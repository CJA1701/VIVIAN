#!/usr/bin/env python3
"""
Vehicle Sentry System Mk3 - Enhanced Person Detection, Facial Recognition & AI Threat Analysis
For Raspberry Pi 5 (8GB) with IMX179 UVC Camera
Python 3.11

Features:
- YOLOv8s person detection with motion pre-filter
- ByteTrack object tracking with persistent IDs
- InsightFace recognition with quality scoring
- Multi-frame confirmation (tolerates 1-2 frame gaps)
- Per-track voting buffers for multi-person support
- AI-powered threat classification using Claude vision
- Known person cooldown (pause, not stop)
- Auto-recovery and graceful degradation
- Log rotation and snapshot management
- Signal handling for clean shutdown
"""

import cv2
import numpy as np
import yaml
import signal
import sys
import os
import traceback
from datetime import datetime, timedelta
import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict, deque
import threading
import queue
import smtplib
import base64
import json
from email.message import EmailMessage
from zoneinfo import ZoneInfo
import requests

# Claude vision runs synchronously on the detection thread, so while it is in
# flight the sentry is blind: no detection, no recording, no deterrent. Keep it
# short — a missed analysis degrades to the pre-rendered deterrent WAV, but a
# blocked analysis leaves the car unguarded.
THREAT_ANALYSIS_TIMEOUT_S = 20.0

# Third-party imports with graceful fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: insightface not installed. Install with: pip install insightface onnxruntime")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Install with: pip install anthropic")


# ============================================================================
# Configuration
# ============================================================================

CONFIG_FILE = Path(__file__).parent / 'sentry_config.yaml'


@dataclass
class SentryConfig:
    """All configurable parameters for the sentry system"""

    # Camera settings
    camera_index: int = 0
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    rotate_180: bool = True
    brightness_mode: str = 'adaptive'
    manual_brightness: float = 1.0

    # Motion detection
    motion_enabled: bool = True
    motion_threshold: int = 25
    motion_min_area: int = 5000
    motion_blur_kernel: int = 21

    # Person detection (YOLO)
    yolo_model: str = 'yolov8n.pt'
    yolo_confidence: float = 0.5
    yolo_iou_threshold: float = 0.45
    yolo_imgsz: int = 640          # inference size; interior subjects are large, 480 is plenty
    person_class_id: int = 0

    # Interior-only filtering
    # Persons inside the cabin appear large; persons outside the windows are
    # small. 0.0 disables the height filter.
    min_person_height_ratio: float = 0.0
    # Fraction of a person bbox that must overlap a deadzone to be discarded
    # (replaces the old center-point-in-zone test, which let through anyone
    # whose bbox center fell outside the zone).
    deadzone_overlap_ratio: float = 0.5

    # Multi-frame confirmation
    confirmation_frames: int = 5
    confirmation_min_hits: int = 4  # Allow 1 gap in 5 frames

    # Object tracking
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    track_buffer: int = 30

    # Face recognition
    face_database_path: str = 'face_database'
    face_similarity_threshold: float = 0.4
    face_voting_frames: int = 7
    face_voting_majority: int = 4
    face_voting_timeout_frames: int = 21  # Give up after this many frames

    # Face quality thresholds
    min_blur_score: float = 100.0
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_face_size: int = 80
    max_face_angle: float = 30.0

    # Known person behavior
    known_person_cooldown: float = 300.0  # 5 min pause after owner recognized
    known_person_stops: bool = False

    # Motion warmup - bypass motion detection for first N seconds
    motion_warmup_seconds: float = 10.0

    # Snapshot management
    snapshot_dir: str = 'snapshots'
    snapshot_known_retention_days: int = 30
    snapshot_unknown_retention_days: int = 30
    # Hard ceiling per snapshot dir, oldest deleted first. Age-based retention
    # cannot bound a busy night, and a full SD card breaks the sentry silently.
    snapshot_max_files: int = 400

    # Logging
    log_dir: str = 'logs'
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5
    detection_log_max_bytes: int = 5 * 1024 * 1024
    detection_log_backup_count: int = 3

    # Camera watchdog
    watchdog_timeout: float = 5.0
    watchdog_max_retries: int = 3
    watchdog_retry_delay: float = 2.0
    watchdog_backoff_multiplier: float = 2.0

    # Alert cooldown
    alert_cooldown_seconds: float = 45.0
    unknown_cooldown_seconds: float = 45.0  # Global cooldown after unknown alert

    # Processing
    process_every_n_frames: int = 2

    # Email alerts
    email_sender: str = ''
    email_app_password: str = ''
    email_recipient: str = ''
    email_smtp_server: str = 'smtp.gmail.com'
    email_smtp_port: int = 587

    # Discord webhook
    discord_webhook_url: str = ''

    # Tamper detection
    tamper_cooldown_seconds: float = 300.0  # 5 min between tamper alerts
    tamper_obstruction_variance_threshold: float = 5.0  # Pixel variance below this = suspicious
    tamper_obstruction_consecutive_seconds: float = 5.0  # Sustained low variance before alert
    tamper_obstruction_check_interval: int = 15  # Check every N frames to save CPU

    # Detection deadzones — list of (x_min, y_min, x_max, y_max) regions to ignore
    deadzones: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Video recording
    recording_dir: str = 'recordings'
    recording_duration_seconds: float = 30.0
    recording_fps: float = 6.0
    recording_codec: str = 'MJPG'
    recording_max_files: int = 35
    recording_enabled: bool = True

    # AI Threat Analysis
    anthropic_api_key: str = ''
    threat_analysis_model: str = 'claude-opus-4-8'
    threat_analysis_enabled: bool = True
    threat_analysis_diagnostic_mode: bool = False
    # Free-text description of the camera's vantage point, injected into the
    # threat prompt so the model places occupants correctly (front/rear,
    # driver/passenger). Empty = no hint.
    threat_analysis_camera_perspective: str = ''

    @classmethod
    def from_yaml(cls, path: Path) -> 'SentryConfig':
        """Load config from YAML file, falling back to defaults for missing keys"""
        if not path.exists():
            print(f"Config file not found: {path} - using defaults")
            return cls()

        with open(path, 'r') as f:
            cfg = yaml.safe_load(f) or {}

        cam = cfg.get('camera', {})
        mot = cfg.get('motion', {})
        det = cfg.get('detection', {})
        trk = cfg.get('tracking', {})
        fr = cfg.get('face_recognition', {})
        fq = cfg.get('face_quality', {})
        kp = cfg.get('known_person', {})
        snap = cfg.get('snapshots', {})
        log = cfg.get('logging', {})
        wd = cfg.get('watchdog', {})
        email = cfg.get('email', {})
        discord = cfg.get('discord', {})
        tamper = cfg.get('tamper_detection', {})
        rec = cfg.get('recording', {})
        ta = cfg.get('threat_analysis', {})

        return cls(
            camera_index=cam.get('index', 0),
            resolution=(cam.get('width', 1920), cam.get('height', 1080)),
            fps=cam.get('fps', 30),
            rotate_180=cam.get('rotate_180', True),
            brightness_mode=cam.get('brightness_mode', 'adaptive'),
            manual_brightness=cam.get('manual_brightness', 1.0),

            motion_enabled=mot.get('enabled', True),
            motion_threshold=mot.get('threshold', 25),
            motion_min_area=mot.get('min_area', 5000),
            motion_blur_kernel=mot.get('blur_kernel', 21),

            yolo_model=det.get('model', 'yolov8n.pt'),
            yolo_confidence=det.get('confidence', 0.5),
            yolo_iou_threshold=det.get('iou_threshold', 0.45),
            yolo_imgsz=det.get('imgsz', 640),
            min_person_height_ratio=det.get('min_person_height_ratio', 0.0),
            deadzone_overlap_ratio=det.get('deadzone_overlap_ratio', 0.5),
            confirmation_frames=det.get('confirmation_frames', 5),
            confirmation_min_hits=det.get('confirmation_min_hits', 4),
            process_every_n_frames=det.get('process_every_n_frames', 2),

            track_high_thresh=trk.get('high_thresh', 0.5),
            track_low_thresh=trk.get('low_thresh', 0.1),
            track_buffer=trk.get('buffer', 30),
            alert_cooldown_seconds=trk.get('alert_cooldown', 45.0),
            unknown_cooldown_seconds=trk.get('unknown_cooldown', 45.0),

            face_database_path=fr.get('database_path', 'face_database'),
            face_similarity_threshold=fr.get('similarity_threshold', 0.4),
            face_voting_frames=fr.get('voting_frames', 7),
            face_voting_majority=fr.get('voting_majority', 4),
            face_voting_timeout_frames=fr.get('voting_timeout_frames', 21),

            min_blur_score=fq.get('min_blur_score', 100.0),
            min_brightness=fq.get('min_brightness', 40.0),
            max_brightness=fq.get('max_brightness', 220.0),
            min_face_size=fq.get('min_face_size', 80),
            max_face_angle=fq.get('max_face_angle', 30.0),

            known_person_cooldown=kp.get('cooldown_seconds', 300.0),
            known_person_stops=kp.get('stop_system', False),
            motion_warmup_seconds=mot.get('warmup_seconds', 10.0),

            snapshot_dir=snap.get('directory', 'snapshots'),
            snapshot_known_retention_days=snap.get('known_retention_days', 30),
            snapshot_unknown_retention_days=snap.get('unknown_retention_days', 30),
            snapshot_max_files=snap.get('max_files', 400),

            log_dir=log.get('directory', 'logs'),
            log_max_bytes=log.get('max_bytes', 10 * 1024 * 1024),
            log_backup_count=log.get('backup_count', 5),
            detection_log_max_bytes=log.get('detection_log_max_bytes', 5 * 1024 * 1024),
            detection_log_backup_count=log.get('detection_log_backup_count', 3),

            watchdog_timeout=wd.get('timeout', 5.0),
            watchdog_max_retries=wd.get('max_retries', 3),
            watchdog_retry_delay=wd.get('retry_delay', 2.0),
            watchdog_backoff_multiplier=wd.get('backoff_multiplier', 2.0),

            email_sender=email.get('sender', ''),
            email_app_password=email.get('app_password', ''),
            email_recipient=email.get('recipient', ''),
            email_smtp_server=email.get('smtp_server', 'smtp.gmail.com'),
            email_smtp_port=email.get('smtp_port', 587),

            discord_webhook_url=discord.get('webhook_url', ''),

            tamper_cooldown_seconds=tamper.get('cooldown_seconds', 300.0),
            tamper_obstruction_variance_threshold=tamper.get('obstruction_variance_threshold', 5.0),
            tamper_obstruction_consecutive_seconds=tamper.get('obstruction_consecutive_seconds', 5.0),
            tamper_obstruction_check_interval=tamper.get('obstruction_check_interval', 15),

            recording_dir=rec.get('directory', 'recordings'),
            recording_duration_seconds=rec.get('duration_seconds', 30.0),
            recording_fps=rec.get('fps', 6.0),
            recording_codec=rec.get('codec', 'MJPG'),
            recording_max_files=rec.get('max_files', 35),
            recording_enabled=rec.get('enabled', True),

            anthropic_api_key=ta.get('anthropic_api_key', ''),
            threat_analysis_model=ta.get('model', 'claude-opus-4-8'),
            threat_analysis_enabled=ta.get('enabled', True),
            threat_analysis_diagnostic_mode=ta.get('diagnostic_mode', False),
            threat_analysis_camera_perspective=ta.get('camera_perspective', ''),

            deadzones=[
                tuple(dz['region']) for dz in det.get('deadzones', [])
                if 'region' in dz and len(dz['region']) == 4
            ],
        )


# ============================================================================
# Logging Setup
# ============================================================================

class LogManager:
    """Manages logging with rotation for all log files"""

    def __init__(self, config: SentryConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Main system logger
        self.logger = logging.getLogger('SentryMk3')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Rotating file handler
        file_handler = RotatingFileHandler(
            self.log_dir / 'sentry_system.log',
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)

        # Detection log with rotation
        self._detection_logger = logging.getLogger('SentryMk3.detections')
        self._detection_logger.setLevel(logging.INFO)
        self._detection_logger.handlers.clear()
        self._detection_logger.propagate = False

        det_handler = RotatingFileHandler(
            self.log_dir / 'detections.log',
            maxBytes=config.detection_log_max_bytes,
            backupCount=config.detection_log_backup_count
        )
        det_handler.setFormatter(logging.Formatter('%(message)s'))
        self._detection_logger.addHandler(det_handler)

    def log_detection(self, person_count: int, confidence: float,
                      identity: str, face_confidence: float, track_ids: List[int]):
        """Log a structured detection event"""
        timestamp = datetime.now().isoformat()
        track_str = ','.join(map(str, track_ids))
        self._detection_logger.info(
            f"{timestamp}|{person_count}|{confidence:.3f}|{identity}|{face_confidence:.3f}|{track_str}"
        )

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)


# ============================================================================
# Motion Detection
# ============================================================================

class MotionDetector:
    """Frame differencing for motion detection pre-filter"""

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.previous_frame: Optional[np.ndarray] = None
        self.motion_detected = False
        self.start_time: Optional[float] = None  # Set when sentry starts
        self._deadzone_mask: Optional[np.ndarray] = None

    def _get_deadzone_mask(self, shape) -> Optional[np.ndarray]:
        """0/1 mask that zeroes the window deadzones, so movement seen
        through the glass (cars, pedestrians) can't wake the YOLO pipeline."""
        if not self.config.deadzones:
            return None
        if self._deadzone_mask is None or self._deadzone_mask.shape != shape[:2]:
            h, w = shape[:2]
            mask = np.ones((h, w), dtype=np.uint8)
            for x1, y1, x2, y2 in self.config.deadzones:
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 0
            self._deadzone_mask = mask
        return self._deadzone_mask

    def detect(self, frame: np.ndarray) -> bool:
        if not self.config.motion_enabled:
            return True

        # Grace period: bypass motion detection for first N seconds
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed < self.config.motion_warmup_seconds:
                return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.config.motion_blur_kernel,) * 2, 0)

        mask = self._get_deadzone_mask(gray.shape)
        if mask is not None:
            gray = gray * mask

        if self.previous_frame is None:
            self.previous_frame = gray
            return False

        frame_delta = cv2.absdiff(self.previous_frame, gray)
        self.previous_frame = gray

        thresh = cv2.threshold(frame_delta, self.config.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            max_area = max(max_area, area)
            if area >= self.config.motion_min_area:
                self.motion_detected = True
                return True

        self.motion_detected = False
        return False

    def reset(self):
        self.previous_frame = None
        self.motion_detected = False


# ============================================================================
# Person Detection with YOLOv8
# ============================================================================

class PersonDetector:
    """YOLOv8 person detection with ByteTrack tracking"""

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.model: Optional[YOLO] = None
        self.confirmation_buffer: deque = deque(maxlen=config.confirmation_frames)
        self.tracked_persons: Dict[int, Dict] = {}
        self.alerted_tracks: Dict[int, datetime] = {}

        self._load_model()

    def _load_model(self):
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - person detection disabled")
            return
        try:
            self.model = YOLO(self.config.yolo_model)
            self.logger.info(f"Loaded YOLO model: {self.config.yolo_model}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        if self.model is None:
            return [], frame

        results = self.model.track(
            frame,
            persist=True,
            conf=self.config.yolo_confidence,
            iou=self.config.yolo_iou_threshold,
            imgsz=self.config.yolo_imgsz,
            classes=[self.config.person_class_id],
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []
        annotated_frame = frame.copy()

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                track_ids = None
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

                for i, (box, conf) in enumerate(zip(boxes, confidences)):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(track_ids[i]) if track_ids is not None else None

                    # Interior-only filters (size + deadzone overlap)
                    if not self._passes_interior_filters((x1, y1, x2, y2), frame.shape):
                        continue

                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'track_id': track_id
                    }
                    detections.append(detection)

                    if track_id is not None:
                        self.tracked_persons[track_id] = {
                            'last_seen': datetime.now(),
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf)
                        }

                    color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"Person {track_id or '?'}: {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return detections, annotated_frame

    def _passes_interior_filters(self, bbox: Tuple[int, int, int, int],
                                 frame_shape) -> bool:
        """Return True if a person bbox looks like it's INSIDE the cabin.

        1. Size filter: interior persons are close to the camera and appear
           large; persons outside the windows are small.
        2. Deadzone overlap: discard when >= deadzone_overlap_ratio of the
           bbox area falls inside any window deadzone. (A center-point test
           let through anyone whose bbox center fell outside the zone.)
        """
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        min_ratio = self.config.min_person_height_ratio
        if min_ratio > 0 and height < min_ratio * frame_shape[0]:
            return False

        bbox_area = width * height
        for dx1, dy1, dx2, dy2 in self.config.deadzones:
            ix = max(0, min(x2, dx2) - max(x1, dx1))
            iy = max(0, min(y2, dy2) - max(y1, dy1))
            if (ix * iy) / bbox_area >= self.config.deadzone_overlap_ratio:
                return False

        return True

    def confirm_detection(self, detections: List[Dict]) -> bool:
        """
        Multi-frame confirmation that tolerates brief gaps.
        Requires confirmation_min_hits out of confirmation_frames.
        """
        has_person = len(detections) > 0
        self.confirmation_buffer.append(has_person)

        if len(self.confirmation_buffer) < self.config.confirmation_frames:
            return False

        hit_count = sum(self.confirmation_buffer)
        return hit_count >= self.config.confirmation_min_hits

    def should_alert(self, track_id: int) -> bool:
        now = datetime.now()
        if track_id in self.alerted_tracks:
            last_alert = self.alerted_tracks[track_id]
            cooldown = timedelta(seconds=self.config.alert_cooldown_seconds)
            if now - last_alert < cooldown:
                return False
        self.alerted_tracks[track_id] = now
        return True

    def get_new_track_ids(self, detections: List[Dict]) -> List[int]:
        new_ids = []
        for det in detections:
            track_id = det.get('track_id')
            if track_id is not None and self.should_alert(track_id):
                new_ids.append(track_id)
        return new_ids

    def cleanup_old_tracks(self, max_age_seconds: float = 60.0):
        now = datetime.now()
        cutoff = timedelta(seconds=max_age_seconds)
        old_tracks = [
            tid for tid, info in self.tracked_persons.items()
            if now - info['last_seen'] > cutoff
        ]
        for tid in old_tracks:
            del self.tracked_persons[tid]
            if tid in self.alerted_tracks:
                del self.alerted_tracks[tid]

    def reset_confirmation(self):
        self.confirmation_buffer.clear()


# ============================================================================
# Face Quality Scoring
# ============================================================================

class FaceQualityScorer:
    """Assess face quality before recognition"""

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger

    def compute_blur_score(self, face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def compute_brightness_score(self, face_img: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)), float(np.std(gray))

    def score_face(self, face_img: np.ndarray,
                   face_bbox: Tuple[int, int, int, int],
                   pose: Optional[Tuple[float, float, float]] = None) -> Dict:
        x1, y1, x2, y2 = face_bbox
        width = x2 - x1
        height = y2 - y1

        size_ok = width >= self.config.min_face_size and height >= self.config.min_face_size

        blur_score = self.compute_blur_score(face_img)
        blur_ok = blur_score >= self.config.min_blur_score

        brightness, contrast = self.compute_brightness_score(face_img)
        brightness_ok = self.config.min_brightness <= brightness <= self.config.max_brightness

        pose_ok = True
        if pose is not None and len(pose) >= 2:
            yaw, pitch = float(pose[0]), float(pose[1])
            pose_ok = (abs(yaw) <= self.config.max_face_angle and
                      abs(pitch) <= self.config.max_face_angle)

        quality_ok = size_ok and blur_ok and brightness_ok and pose_ok

        return {
            'quality_ok': quality_ok,
            'size': (width, height),
            'size_ok': size_ok,
            'blur_score': blur_score,
            'blur_ok': blur_ok,
            'brightness': brightness,
            'contrast': contrast,
            'brightness_ok': brightness_ok,
            'pose': pose,
            'pose_ok': pose_ok
        }


# ============================================================================
# Face Recognition with InsightFace (multi-person, per-track voting)
# ============================================================================

class FaceRecognizer:
    """InsightFace-based face recognition with per-track voting"""

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.app: Optional[FaceAnalysis] = None
        self.known_faces: Dict[str, List[np.ndarray]] = {}
        self.quality_scorer = FaceQualityScorer(config, logger)

        # Per-track voting: track_id -> deque of vote dicts
        self.track_voting: Dict[int, deque] = {}
        # Per-track frame counters for timeout
        self.track_vote_frames: Dict[int, int] = {}
        # Consecutive frames a voting track has been absent from detections,
        # used to evict buffers for tracks ByteTrack has re-IDed away.
        self.track_missing_frames: Dict[int, int] = {}

        self._initialize()

    def _initialize(self):
        if not INSIGHTFACE_AVAILABLE:
            self.logger.error("InsightFace not available - face recognition disabled")
            return
        try:
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            # Faces are detected inside person-bbox crops (close-range,
            # large faces), so a small det_size is sufficient and ~4x
            # cheaper than the old full-frame 640x640 pass.
            self.app.prepare(ctx_id=-1, det_size=(320, 320))
            self.logger.info("InsightFace initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize InsightFace: {e}")
            self.app = None
            return
        self._load_known_faces()

    def _load_known_faces(self):
        db_path = Path(self.config.face_database_path)
        db_path.mkdir(parents=True, exist_ok=True)

        if self.app is None:
            return

        image_extensions = ['*.jpg', '*.jpeg', '*.png']

        # Subfolder structure: face_database/PersonName/*.jpg
        for person_dir in db_path.iterdir():
            if person_dir.is_dir() and not person_dir.name.startswith('.'):
                person_name = person_dir.name
                self.known_faces[person_name] = []
                for ext in image_extensions:
                    for img_path in person_dir.glob(ext):
                        self._load_face_image(img_path, person_name)

        # Flat structure: face_database/PersonName.jpg
        for ext in image_extensions:
            for img_path in db_path.glob(ext):
                if img_path.is_file():
                    person_name = img_path.stem
                    if person_name not in self.known_faces:
                        self.known_faces[person_name] = []
                    self._load_face_image(img_path, person_name)

        total_faces = sum(len(embs) for embs in self.known_faces.values())
        self.logger.info(f"Loaded {total_faces} face encodings for {len(self.known_faces)} persons")

    def _load_face_image(self, img_path: Path, person_name: str):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                self.logger.warning(f"Could not read image: {img_path}")
                return
            faces = self.app.get(img)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                self.known_faces[person_name].append(face.embedding)
                self.logger.info(f"Loaded face: {person_name} from {img_path.name}")
            else:
                self.logger.warning(f"No face found in {img_path}")
        except Exception as e:
            self.logger.error(f"Error loading {img_path}: {e}")

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        if self.app is None:
            return []
        try:
            faces = self.app.get(frame)
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return []

        face_infos = []
        for face in faces:
            bbox = tuple(map(int, face.bbox))
            x1, y1, x2, y2 = bbox

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_img = frame[y1:y2, x1:x2]

            # Get pose directly from InsightFace
            pose = None
            if hasattr(face, 'pose'):
                pose = tuple(float(v) for v in face.pose)

            quality = self.quality_scorer.score_face(face_img, (x1, y1, x2, y2), pose)

            face_infos.append({
                'bbox': (x1, y1, x2, y2),
                'embedding': face.embedding,
                'quality': quality,
                'pose': pose,
                'face_img': face_img
            })

        return face_infos

    def identify_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        if not self.known_faces:
            return 'Unknown', 0.0

        best_match = 'Unknown'
        best_similarity = 0.0

        for name, embeddings in self.known_faces.items():
            for known_emb in embeddings:
                similarity = np.dot(embedding, known_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_emb)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= self.config.face_similarity_threshold:
                        best_match = name

        return best_match, float(best_similarity)

    def _detect_face_in_crop(self, frame: np.ndarray,
                             person_bbox: Tuple[int, int, int, int]) -> Optional[Dict]:
        """Run face detection on a person-bbox crop (much cheaper than the
        full frame) and return the largest face, or None."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = person_bbox
        # Margin so faces at the bbox edge aren't clipped
        mx = int((x2 - x1) * 0.15)
        my = int((y2 - y1) * 0.10)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            return None

        try:
            faces = self.app.get(crop)
        except Exception as e:
            self.logger.error(f"Face detection error: {e}")
            return None
        if not faces:
            return None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(crop.shape[1], fx2), min(crop.shape[0], fy2)
        if fx2 <= fx1 or fy2 <= fy1:
            return None

        face_img = crop[fy1:fy2, fx1:fx2]
        pose = None
        if hasattr(face, 'pose'):
            pose = tuple(float(v) for v in face.pose)
        quality = self.quality_scorer.score_face(face_img, (fx1, fy1, fx2, fy2), pose)

        return {
            'embedding': face.embedding,
            'quality': quality,
            'pose': pose,
        }

    def identify_all_persons(self, frame: np.ndarray,
                             person_detections: List[Dict]) -> Optional[Dict]:
        """
        Identify faces (per person-bbox crop) using per-track voting buffers.

        Returns:
            Dict with results when ANY track reaches voting majority, or None if still collecting.
            Result keys: known_names, unknown_track_ids, has_known, all_results
        """
        # Track which person tracks have faces this frame
        tracks_with_faces = set()

        for det in person_detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue

            face_info = self._detect_face_in_crop(frame, det['bbox'])
            if face_info is None:
                continue

            tracks_with_faces.add(track_id)

            # Pick quality faces or fall back to best available
            if not face_info['quality']['quality_ok']:
                # Still use it but note low quality
                self.logger.debug(f"Low quality face for track {track_id}")

            name, similarity = self.identify_face(face_info['embedding'])

            # Initialize per-track voting buffer if needed
            if track_id not in self.track_voting:
                self.track_voting[track_id] = deque(maxlen=self.config.face_voting_frames)
                self.track_vote_frames[track_id] = 0

            self.track_voting[track_id].append({
                'name': name,
                'similarity': similarity,
                'quality': face_info['quality']
            })

        # Increment frame counters for all active tracks
        active_track_ids = [d.get('track_id') for d in person_detections if d.get('track_id') is not None]
        for tid in active_track_ids:
            if tid in self.track_vote_frames:
                self.track_vote_frames[tid] += 1

        # Evict vote buffers for tracks that are no longer being detected.
        # ByteTrack re-IDs a person after an occlusion (id 5 becomes id 9) and
        # the abandoned id's buffer used to live forever: track_vote_frames only
        # advances for tids present in the current frame, so the abandoned
        # entry's timeout could never fire. A lingering entry keeps
        # has_any_votes True in process_frame, which zeroes the
        # confirmed-no-face counter every frame — permanently disabling the only
        # alert path for an intruder whose face never resolves.
        active_set = set(active_track_ids)
        for tid in list(self.track_voting.keys()):
            if tid in active_set:
                self.track_missing_frames.pop(tid, None)
                continue
            missed = self.track_missing_frames.get(tid, 0) + 1
            self.track_missing_frames[tid] = missed
            if missed >= self.config.face_voting_timeout_frames:
                self.logger.debug(
                    f"Evicting stale vote buffer for track {tid} "
                    f"(absent {missed} frames)"
                )
                self.track_voting.pop(tid, None)
                self.track_vote_frames.pop(tid, None)
                self.track_missing_frames.pop(tid, None)

        # Check if any tracks have reached voting threshold or timed out
        resolved_tracks: Dict[int, Dict] = {}  # track_id -> result
        timed_out_tracks: List[int] = []

        for tid in list(self.track_voting.keys()):
            votes = self.track_voting[tid]
            frames_elapsed = self.track_vote_frames.get(tid, 0)

            # Try to resolve first, so a track that has both a full buffer and
            # a clear majority resolves normally even on its timeout frame.
            if len(votes) >= self.config.face_voting_frames:
                # Tally votes
                vote_counts: Dict[str, int] = defaultdict(int)
                vote_sims: Dict[str, List[float]] = defaultdict(list)
                for vote in votes:
                    vote_counts[vote['name']] += 1
                    vote_sims[vote['name']].append(vote['similarity'])

                winner_name, winner_votes = max(vote_counts.items(), key=lambda x: x[1])

                if winner_votes >= self.config.face_voting_majority:
                    avg_sim = float(np.mean(vote_sims[winner_name]))
                    resolved_tracks[tid] = {
                        'name': winner_name,
                        'identified': winner_name != 'Unknown',
                        'confidence': avg_sim,
                        'votes': winner_votes,
                        'total_frames': self.config.face_voting_frames
                    }
                    continue

            # Couldn't resolve (too few votes, or a full buffer with no
            # majority). Time out once we've waited long enough. The old check
            # also required len(votes) < face_voting_frames, which meant a FULL
            # buffer that never reached a majority could never time out — its
            # entry then lived forever, and because a lingering entry keeps
            # has_any_votes True it permanently disabled the confirmed-no-face
            # alert path for anyone whose face never resolves.
            if frames_elapsed >= self.config.face_voting_timeout_frames:
                timed_out_tracks.append(tid)

        # Clean up timed-out tracks (treat as no-face / unknown)
        for tid in timed_out_tracks:
            resolved_tracks[tid] = {
                'name': 'Unknown',
                'identified': False,
                'confidence': 0.0,
                'votes': 0,
                'total_frames': self.track_vote_frames.get(tid, 0),
                'timed_out': True
            }

        if not resolved_tracks:
            return None

        # Clean up resolved track buffers
        for tid in resolved_tracks:
            self.track_voting.pop(tid, None)
            self.track_vote_frames.pop(tid, None)
            self.track_missing_frames.pop(tid, None)

        # Categorize results
        known_names = []
        unknown_track_ids = []
        for tid, result in resolved_tracks.items():
            if result['identified']:
                known_names.append(result['name'])
            else:
                unknown_track_ids.append(tid)

        return {
            'has_known': len(known_names) > 0,
            'known_names': known_names,
            'unknown_track_ids': unknown_track_ids,
            'all_results': resolved_tracks
        }

    def reset_voting(self, track_id: Optional[int] = None):
        """Clear voting buffers. If track_id given, only that track."""
        if track_id is not None:
            self.track_voting.pop(track_id, None)
            self.track_vote_frames.pop(track_id, None)
            self.track_missing_frames.pop(track_id, None)
        else:
            self.track_voting.clear()
            self.track_vote_frames.clear()
            self.track_missing_frames.clear()

    def reload_database(self):
        self.known_faces.clear()
        self._load_known_faces()


# ============================================================================
# Snapshot Management
# ============================================================================

class SnapshotManager:
    """Manages snapshot storage and cleanup"""

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.snapshot_dir = Path(config.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.known_dir = self.snapshot_dir / 'known'
        self.unknown_dir = self.snapshot_dir / 'unknown'
        self.known_dir.mkdir(exist_ok=True)
        self.unknown_dir.mkdir(exist_ok=True)

    def save_snapshot(self, frame: np.ndarray, person_count: int,
                      is_known: bool = False, person_name: str = '') -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        if is_known and person_name:
            subdir = self.known_dir / person_name
            subdir.mkdir(exist_ok=True)
            filename = subdir / f"{timestamp}_{person_count}p.jpg"
        else:
            filename = self.unknown_dir / f"detection_{timestamp}_{person_count}p.jpg"

        # Check the return value: on a full SD card imwrite fails silently, and
        # the caller then hands a non-existent path to the threat analyzer, whose
        # encode step raises and is swallowed into a bogus level-5 "analysis
        # unavailable". A loud log here is the difference between "disk is full"
        # and "the AI is broken".
        if not cv2.imwrite(str(filename), frame):
            self.logger.error(
                f"Snapshot write FAILED (disk full or unwritable?): {filename}"
            )
        else:
            self.logger.info(f"Snapshot saved: {filename}")
        return filename

    def cleanup_old_snapshots(self):
        now = datetime.now()

        unknown_cutoff = now - timedelta(days=self.config.snapshot_unknown_retention_days)
        self._cleanup_directory(self.unknown_dir, unknown_cutoff)

        known_cutoff = now - timedelta(days=self.config.snapshot_known_retention_days)
        self._cleanup_directory(self.known_dir, known_cutoff, recursive=True)

        # Legacy flat snapshots
        legacy_cutoff = now - timedelta(days=self.config.snapshot_unknown_retention_days)
        self._cleanup_directory(self.snapshot_dir, legacy_cutoff, recursive=False)

        # Age-based retention alone cannot bound disk usage: a night of repeated
        # triggers writes hundreds of JPEGs that are all well inside the
        # retention window. Recordings are capped by file count; snapshots were
        # not, so a full SD card silently broke snapshotting AND threat analysis
        # (a missing file makes _encode_image_base64 raise, which the analyzer
        # swallows into a bogus level-5 "analysis unavailable").
        self._enforce_count_cap(self.unknown_dir, recursive=False)
        self._enforce_count_cap(self.known_dir, recursive=True)

    def _enforce_count_cap(self, directory: Path, recursive: bool = False):
        """Delete oldest-first so a directory never exceeds the file cap."""
        cap = self.config.snapshot_max_files
        if cap <= 0 or not directory.exists():
            return
        pattern = '**/*.jpg' if recursive else '*.jpg'
        try:
            images = sorted(
                (p for p in directory.glob(pattern) if p.is_file()),
                key=lambda p: p.stat().st_mtime
            )
            excess = len(images) - cap
            for old in images[:excess] if excess > 0 else []:
                old.unlink()
            if excess > 0:
                self.logger.info(
                    f"Snapshot cap ({cap}): deleted {excess} oldest from {directory}"
                )
        except Exception as e:
            self.logger.error(f"Snapshot cap enforcement failed for {directory}: {e}")

    def _cleanup_directory(self, directory: Path, cutoff: datetime, recursive: bool = False):
        if not directory.exists():
            return
        pattern = '**/*.jpg' if recursive else '*.jpg'
        deleted = 0
        for img_path in directory.glob(pattern):
            if img_path.is_file():
                try:
                    mtime = datetime.fromtimestamp(img_path.stat().st_mtime)
                    if mtime < cutoff:
                        img_path.unlink()
                        deleted += 1
                except Exception as e:
                    self.logger.error(f"Error deleting {img_path}: {e}")
        if deleted > 0:
            self.logger.info(f"Cleaned up {deleted} old snapshots from {directory}")


# ============================================================================
# Video Recorder
# ============================================================================

class VideoRecorder:
    """Records video clips on detection events using a background writer thread.

    Uses MJPG/.avi for crash-safe files (no moov atom issues like MP4).
    Only one recording at a time; subsequent start_recording calls are
    rejected while a recording is active.
    """

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.recording_dir = Path(config.recording_dir)
        self.recording_dir.mkdir(parents=True, exist_ok=True)

        self._recording = False
        self._frame_queue: Optional[queue.Queue] = None
        self._stop_event = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._recording_start: Optional[float] = None
        self._frame_size: Optional[Tuple[int, int]] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start_recording(self, trigger_type: str) -> bool:
        """Begin a new recording. Returns False if already recording."""
        if self._recording:
            self.logger.info(f"Recording already in progress, ignoring {trigger_type} trigger")
            return False

        # A previous writer may still be draining. It must be finished before we
        # rebind state, otherwise it wakes up on the NEW queue and writes this
        # recording's frames into the PREVIOUS file (and the old stop_event
        # signal is erased by clear()), splitting both clips in half.
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5)
            if self._writer_thread.is_alive():
                self.logger.error(
                    "Previous recording writer still running — refusing to "
                    f"start {trigger_type} recording"
                )
                return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{trigger_type}_{timestamp}.avi"
        filepath = self.recording_dir / filename

        # Per-recording queue and stop event, handed to the thread as arguments
        # rather than shared through instance attributes, so a lingering writer
        # can never observe the next recording's state.
        frame_queue: "queue.Queue" = queue.Queue(maxsize=300)
        stop_event = threading.Event()
        self._frame_queue = frame_queue
        self._stop_event = stop_event
        self._recording = True
        # monotonic: wall clock steps at boot (no RTC battery), which could make
        # this elapsed check negative or enormous.
        self._recording_start = time.monotonic()
        self._frame_size = None

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(filepath, frame_queue, stop_event),
            daemon=True
        )
        self._writer_thread.start()
        self.logger.info(f"Recording started: {filepath}")
        return True

    def poll(self) -> None:
        """Enforce the duration limit even when no frames are arriving.

        The auto-stop used to live ONLY in feed_frame(), so when the camera
        stalled — or the run loop's stale-frame path skipped feeding — the
        recording latched on forever: the writer never saw the stop event and
        every later start_recording() was refused, killing video capture for the
        rest of the sentry run, precisely during the event it exists to record.
        """
        if not self._recording or self._recording_start is None:
            return
        if time.monotonic() - self._recording_start >= self.config.recording_duration_seconds:
            self.stop_recording()

    def feed_frame(self, frame: np.ndarray) -> None:
        """Feed a frame to the active recording. Auto-stops after duration limit."""
        if not self._recording or self._frame_queue is None:
            return

        # Auto-stop after configured duration
        if time.monotonic() - self._recording_start >= self.config.recording_duration_seconds:
            self.stop_recording()
            return

        # Capture frame size from first frame
        if self._frame_size is None:
            self._frame_size = (frame.shape[1], frame.shape[0])

        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Drop frame rather than block the main loop

    def stop_recording(self) -> None:
        """Signal the writer thread to finish and stop recording."""
        if not self._recording:
            return
        self._recording = False
        self._stop_event.set()
        self.logger.info("Recording stop signalled")

    def _writer_loop(self, filepath: Path, frame_queue: "queue.Queue",
                     stop_event: threading.Event) -> None:
        """Background thread: drains frame queue and writes to video file.

        The queue and stop event are parameters, not instance attributes, so
        this thread is bound to exactly one recording for its whole life.
        """
        writer = None
        frames_written = 0
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.config.recording_codec)

            while not stop_event.is_set() or not frame_queue.empty():
                try:
                    frame = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Lazy-init writer on first frame (need frame dimensions)
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(filepath), fourcc,
                        self.config.recording_fps, (w, h)
                    )
                    if not writer.isOpened():
                        self.logger.error(f"Failed to open VideoWriter for {filepath}")
                        break

                writer.write(frame)
                frames_written += 1

        except Exception as e:
            self.logger.error(f"Recording writer error: {e}")
        finally:
            if writer is not None:
                writer.release()
            self.logger.info(
                f"Recording saved: {filepath} ({frames_written} frames, "
                f"{frames_written / max(self.config.recording_fps, 1):.1f}s)"
            )
            self._cleanup_old_recordings()

    def _cleanup_old_recordings(self) -> None:
        """Delete oldest .avi files beyond max_files limit."""
        try:
            recordings = sorted(
                self.recording_dir.glob('*.avi'),
                key=lambda p: p.stat().st_mtime
            )
            excess = len(recordings) - self.config.recording_max_files
            if excess > 0:
                for old_file in recordings[:excess]:
                    old_file.unlink()
                    self.logger.info(f"Deleted old recording: {old_file.name}")
        except Exception as e:
            self.logger.error(f"Recording cleanup error: {e}")

    def release(self) -> None:
        """Force-stop recording and wait for writer thread to finish."""
        self.stop_recording()
        if self._writer_thread is not None and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=10)
            # Only forget the thread if it actually finished — clearing the
            # reference on a still-running writer orphans it along with its open
            # VideoWriter handle.
            if self._writer_thread.is_alive():
                self.logger.error(
                    "Recording writer did not exit within 10s — clip may be truncated"
                )
            else:
                self._writer_thread = None


# ============================================================================
# AI Threat Analyzer
# ============================================================================

@dataclass
class ThreatAssessment:
    """Result of threat analysis"""
    threat_level: int  # 1-10 scale
    description: str
    reasoning: str
    deterrent_message: str = ''  # Spoken aloud near the vehicle as a warning
    error: Optional[str] = None


class ThreatAnalyzer:
    """AI-powered threat classification using Claude vision.

    Model is set via sentry_config.yaml threat_analysis.model
    (default claude-opus-4-8). Must support vision + structured outputs
    (Opus 4.8, Sonnet 5, or Haiku 4.5).
    """

    # Structured output schema — the API guarantees the response is valid
    # JSON matching this shape, so no text-recovery parsing is needed.
    OUTPUT_SCHEMA = {
        "type": "object",
        "properties": {
            "activity_level": {
                "type": "integer",
                "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "description": "Activity/concern rating"
            },
            "description": {
                "type": "string",
                "description": "What is observed, max 15 words"
            },
            "reasoning": {
                "type": "string",
                "description": "1-2 sentence explanation"
            },
            "deterrent_message": {
                "type": "string",
                "description": ("2-3 sentence spoken notice specific to the observed "
                                "action/object; empty in diagnostic mode")
            }
        },
        "required": ["activity_level", "description", "reasoning", "deterrent_message"],
        "additionalProperties": False
    }

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.client = None

        self._initialize()

    def _initialize(self):
        if not ANTHROPIC_AVAILABLE:
            self.logger.warning("Anthropic library not available - threat analysis disabled")
            return

        if not self.config.anthropic_api_key:
            self.logger.warning("Anthropic API key not configured - threat analysis disabled")
            return

        try:
            # Explicit short timeout + single retry. The SDK default is a 600s
            # timeout with 2 retries, so ONE vision call in a weak-signal spot
            # (parked in a garage) could block the detection thread for ~30
            # minutes: no detection, no recording, no deterrent, and no alert —
            # the car sat unguarded for the whole window.
            self.client = anthropic.Anthropic(
                api_key=self.config.anthropic_api_key,
                timeout=THREAT_ANALYSIS_TIMEOUT_S,
                max_retries=1,
            )
            self.logger.info("Claude threat analyzer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            self.client = None

    def _encode_image_base64(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def analyze_threat(self, snapshot_path: Path) -> ThreatAssessment:
        """Analyze a snapshot for threat level using Claude vision."""
        if self.client is None or not self.config.threat_analysis_enabled:
            return ThreatAssessment(
                threat_level=5,
                description="Unknown - threat analysis disabled",
                reasoning="Threat analysis not configured",
                error="Anthropic not available"
            )

        try:
            base64_image = self._encode_image_base64(snapshot_path)

            system_msg = (
                "You are a vehicle monitoring assistant helping a car owner review interior camera footage. "
                "The owner has set up an automated dashcam system in their personal vehicle to notify them "
                "of activity while it is parked. You describe what the camera sees so the owner can review it remotely."
            )

            if self.config.threat_analysis_diagnostic_mode:
                prompt = (
                    "This image is from the owner's parked vehicle interior camera. "
                    "Describe what you see — is anyone present, what are they doing, anything noteworthy? "
                    "Keep the reasoning to 2-3 sentences. Set activity_level to 0, description to "
                    "'Diagnostic Test - Analysis Working', and deterrent_message to an empty string."
                )
            else:
                perspective = self.config.threat_analysis_camera_perspective.strip()
                perspective_block = f"Camera placement: {perspective}\n\n" if perspective else ""
                prompt = (
                    "This image is from the interior camera of a privately owned parked vehicle. "
                    "The owner wants to know what is happening inside their car while it is parked.\n\n"
                    f"{perspective_block}"
                    "Look closely at the person's posture and hands, and at any objects or tools they "
                    "are holding or reaching for — e.g. a screwdriver, pry bar, cutting tool, wires, "
                    "the ignition/steering column, or open storage compartments. Rate the activity "
                    "level from 1 to 10 by how unusual or concerning it is to the owner:\n"
                    "1-2: Normal or expected — empty interior, or the owner/a known person acting routinely.\n"
                    "3-4: Unfamiliar person present but idle or just seated; not touching anything.\n"
                    "5-6: Person is searching, opening compartments, or handling items that aren't theirs.\n"
                    "7-8: Person is actively going through the vehicle or handling a tool near the controls.\n"
                    "9-10: Person is attempting to start, hotwire, force, or damage the vehicle, or holds a "
                    "tool at the ignition, steering column, or a window.\n\n"
                    "Then write a short spoken notice (2-3 sentences, under 40 words, second person "
                    "\"you\") to play through the car's speakers. Make it SPECIFIC to what you actually "
                    "see — name the item in their hands or the exact action in progress instead of a "
                    "generic warning. Escalate with the level:\n"
                    "- Low (1-3): Polite — the vehicle is monitored and the owner has been notified.\n"
                    "- Medium (4-6): Firm — reference what they are doing; it is being recorded and "
                    "reported to the owner.\n"
                    "- High (7-10): A direct command to stop the specific action and exit now; their "
                    "face and biometric data have been logged and the owner has been alerted.\n\n"
                    "Example (high, screwdriver at the steering column): \"Put down the screwdriver and "
                    "step out of the vehicle now. Your face and biometrics have been logged, and the "
                    "owner has been alerted.\"\n"
                    "Base everything only on what is actually visible — do not invent objects or actions. "
                    "Do not reference any AI system by name."
                )

            self.logger.info("Analyzing threat with Claude vision...")

            response = self.client.messages.create(
                model=self.config.threat_analysis_model,
                max_tokens=500,
                system=system_msg,
                output_config={"format": {"type": "json_schema", "schema": self.OUTPUT_SCHEMA}},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image,
                                }
                            },
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
            )

            # Safety classifiers can decline; fall back to the static WAV
            # deterrent path exactly like any other analysis failure.
            if response.stop_reason == "refusal":
                self.logger.warning("Claude declined to analyze the snapshot (refusal)")
                return ThreatAssessment(
                    threat_level=5,
                    description="Analysis unavailable",
                    reasoning="Model declined to analyze this image",
                    error="Refusal"
                )

            result_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            ).strip()
            if not result_text:
                self.logger.warning(
                    f"Claude returned empty content — stop_reason={response.stop_reason!r}"
                )
                return ThreatAssessment(
                    threat_level=5,
                    description="Analysis unavailable",
                    reasoning=f"Model did not respond (stop_reason={response.stop_reason})",
                    error="Empty response"
                )

            # Structured outputs guarantee valid JSON matching OUTPUT_SCHEMA
            result = json.loads(result_text)

            assessment = ThreatAssessment(
                threat_level=int(result['activity_level']),
                description=result['description'],
                reasoning=result['reasoning'],
                deterrent_message=result.get('deterrent_message', '')
            )

            self.logger.info(f"Threat analysis complete: Level {assessment.threat_level}/10 - {assessment.description}")
            return assessment

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Claude response as JSON: {e}")
            return ThreatAssessment(
                threat_level=5,
                description="Analysis error",
                reasoning="Failed to parse AI response",
                error=str(e)
            )
        except Exception as e:
            self.logger.error(f"Threat analysis failed: {e}")
            return ThreatAssessment(
                threat_level=5,
                description="Analysis unavailable",
                reasoning="Error during analysis",
                error=str(e)
            )


# ============================================================================
# Camera Watchdog
# ============================================================================

class CameraWatchdog:
    """Monitors camera health and handles recovery"""

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self.last_frame_time: Optional[datetime] = None
        self.consecutive_failures: int = 0
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_healthy: bool = False
        self._lock = threading.Lock()

    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        retry_delay = self.config.watchdog_retry_delay

        for attempt in range(self.config.watchdog_max_retries):
            try:
                self.logger.info(f"Camera init attempt {attempt + 1}/{self.config.watchdog_max_retries}")

                camera = cv2.VideoCapture(self.config.camera_index)
                if not camera.isOpened():
                    raise RuntimeError(f"Failed to open camera at index {self.config.camera_index}")

                camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
                camera.set(cv2.CAP_PROP_FPS, self.config.fps)
                camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

                ret, _ = camera.read()
                if not ret:
                    camera.release()
                    raise RuntimeError("Camera opened but failed to read frame")

                actual_w = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_h = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = camera.get(cv2.CAP_PROP_FPS)
                self.logger.info(f"Camera initialized: {actual_w}x{actual_h} @ {actual_fps}fps")

                with self._lock:
                    self.camera = camera
                    self.is_healthy = True
                    self.consecutive_failures = 0
                    self.last_frame_time = datetime.now()

                return camera

            except Exception as e:
                self.logger.error(f"Camera init failed: {e}")
                if attempt < self.config.watchdog_max_retries - 1:
                    self.logger.info(f"Retrying in {retry_delay:.1f}s...")
                    time.sleep(retry_delay)
                    retry_delay *= self.config.watchdog_backoff_multiplier

        self.logger.error("Camera initialization failed after all retries")
        return None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        # The blocking read MUST happen outside the lock. camera.read() has no
        # timeout — V4L2's VIDIOC_DQBUF can stall indefinitely on a USB bus
        # reset while the fd stays valid — and attempt_recovery()/check_health()
        # need this same lock. Holding it across the read deadlocked recovery
        # permanently: the sentry never recovered, activate() never returned,
        # VIVIAN never resumed wake-word listening, and systemd never restarted
        # (the process was alive). That was a power-cycle-only failure.
        with self._lock:
            if self.camera is None or not self.is_healthy:
                return False, None
            camera = self.camera

        ret, frame = camera.read()

        with self._lock:
            # Recovery may have released/replaced the camera while we were
            # blocked in read() — discard results from the stale handle.
            if camera is not self.camera:
                return False, None

            if ret:
                self.last_frame_time = datetime.now()
                self.consecutive_failures = 0
                return True, frame
            else:
                self.consecutive_failures += 1
                self.logger.warning(f"Frame read failed ({self.consecutive_failures} consecutive)")
                if self.consecutive_failures >= 3:
                    self.is_healthy = False
                return False, None

    def check_health(self) -> bool:
        with self._lock:
            if not self.is_healthy or self.last_frame_time is None:
                return False
            elapsed = (datetime.now() - self.last_frame_time).total_seconds()
            if elapsed > self.config.watchdog_timeout:
                self.logger.warning(f"Camera timeout: {elapsed:.1f}s since last frame")
                self.is_healthy = False
                return False
            return True

    def attempt_recovery(self) -> bool:
        self.logger.info("Attempting camera recovery...")
        with self._lock:
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception:
                    pass
                self.camera = None
            self.is_healthy = False

        time.sleep(self.config.watchdog_retry_delay)
        return self.initialize_camera() is not None

    def release(self):
        with self._lock:
            if self.camera is not None:
                try:
                    self.camera.release()
                except Exception:
                    pass
                self.camera = None
            self.is_healthy = False


# ============================================================================
# Tamper Detection
# ============================================================================

class TamperDetector:
    """Detects camera obstruction via sustained low pixel variance.

    Low variance catches both dark covers (tape, hand) and bright covers
    (white cloth, paper) since both produce near-uniform images.  Real
    nighttime scenes still have sensor noise (variance 10-50+), so a
    threshold of 5.0 is well below normal.  The 5-second sustained
    duration prevents false positives from momentary occlusions.
    """

    def __init__(self, config: SentryConfig, logger: LogManager):
        self.config = config
        self.logger = logger
        self._low_variance_start: Optional[float] = None

    def check_frame(self, frame: np.ndarray) -> bool:
        """Return True if obstruction is confirmed (sustained low variance)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = float(np.var(gray))

        if variance < self.config.tamper_obstruction_variance_threshold:
            now = time.time()
            if self._low_variance_start is None:
                self._low_variance_start = now
                self.logger.warning(
                    f"Tamper: low variance detected ({variance:.2f}), starting timer"
                )
            elif now - self._low_variance_start >= self.config.tamper_obstruction_consecutive_seconds:
                self.logger.warning(
                    f"Tamper: sustained low variance ({variance:.2f}) for "
                    f"{now - self._low_variance_start:.1f}s - obstruction confirmed"
                )
                return True
        else:
            if self._low_variance_start is not None:
                self.logger.info("Tamper: variance recovered, resetting timer")
            self._low_variance_start = None

        return False

    def reset(self):
        """Clear timer state (e.g. after camera recovery)."""
        self._low_variance_start = None


# ============================================================================
# Main Sentry System
# ============================================================================

class VehicleSentry:
    """Main sentry system orchestrator"""

    def __init__(self, config: Optional[SentryConfig] = None):
        self.config = config or SentryConfig()

        # Initialize components
        self.logger = LogManager(self.config)
        self.logger.info("Initializing Vehicle Sentry System Mk3...")

        self.motion_detector = MotionDetector(self.config, self.logger)
        self.person_detector = PersonDetector(self.config, self.logger)
        self.snapshot_manager = SnapshotManager(self.config, self.logger)
        self.watchdog = CameraWatchdog(self.config, self.logger)
        self.threat_analyzer = ThreatAnalyzer(self.config, self.logger)
        self.tamper_detector = TamperDetector(self.config, self.logger)

        # Video recorder
        self.video_recorder: Optional[VideoRecorder] = None
        if self.config.recording_enabled:
            self.video_recorder = VideoRecorder(self.config, self.logger)

        # Face recognizer (may fail gracefully)
        self.face_recognizer: Optional[FaceRecognizer] = None
        self._init_face_recognizer()

        # State
        self.running = False
        self.paused_until: Optional[datetime] = None  # Known-person cooldown
        self.unknown_paused_until: Optional[datetime] = None  # Unknown-person cooldown
        self.frame_count = 0
        self.last_cleanup = datetime.now()
        self._confirmed_no_face_frames = 0  # Frames since confirmation with no face votes
        self._last_tamper_alert: Optional[datetime] = None

    def _init_face_recognizer(self):
        try:
            self.face_recognizer = FaceRecognizer(self.config, self.logger)
            if self.face_recognizer.app is None:
                self.logger.warning("Face recognition unavailable - will alert on all persons")
                self.face_recognizer = None
        except Exception as e:
            self.logger.error(f"Face recognizer init failed: {e}")
            self.face_recognizer = None

    def adjust_brightness(self, frame: np.ndarray) -> np.ndarray:
        if self.config.brightness_mode == 'off':
            return frame
        elif self.config.brightness_mode == 'manual':
            return cv2.convertScaleAbs(frame, alpha=self.config.manual_brightness, beta=0)
        elif self.config.brightness_mode == 'adaptive':
            # CLAHE only helps badly-exposed scenes (night, glare). Skip it
            # when the frame is already well exposed — it's one of the most
            # expensive per-frame steps at 720p.
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            mean_luma = float(small.mean())
            if 60.0 <= mean_luma <= 190.0:
                return frame
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return frame

    def handle_known_person(self, names: List[str], confidence: float, frame: np.ndarray):
        """Handle known person(s) detected - pause monitoring instead of stopping"""
        names_str = ', '.join(n.upper() for n in names)
        self.logger.info(f"KNOWN PERSON(S): {names_str} (confidence: {confidence:.2f})")

        person_count = len(self.person_detector.tracked_persons)
        for name in names:
            self.snapshot_manager.save_snapshot(frame, person_count, is_known=True, person_name=name)

        # Pause sentry for cooldown period instead of stopping
        self.paused_until = datetime.now() + timedelta(seconds=self.config.known_person_cooldown)
        self.logger.info(f"Sentry paused for {self.config.known_person_cooldown}s")

    def _send_alerts_async(self, snapshot_path, person_count, threat):
        """Fire email + Discord alerts on a background thread so SMTP/Discord
        latency (seconds over cellular) doesn't pause the detection loop."""
        def _send():
            try:
                self._send_alert_email(snapshot_path, person_count, threat)
                self._send_discord_alert(snapshot_path, person_count, threat)
            except Exception as e:
                self.logger.error(f"Async alert send failed: {e}")
        threading.Thread(target=_send, daemon=True).start()

    def handle_unknown_person(self, frame: np.ndarray, track_ids: List[int]):
        """Handle unknown person(s) detected"""
        self.logger.warning(f"ALERT: UNKNOWN PERSON(S) DETECTED! (tracks: {track_ids})")

        # Start video recording FIRST — threat analysis + email + Discord
        # take seconds over cellular, and those are the seconds that matter.
        if self.video_recorder is not None:
            self.video_recorder.start_recording('unknown')

        person_count = len(track_ids)
        snapshot_path = self.snapshot_manager.save_snapshot(frame, person_count, is_known=False)

        # Run AI threat analysis on the snapshot
        threat = self.threat_analyzer.analyze_threat(snapshot_path)

        # Email + Discord fire in the background (don't block detection)
        self._send_alerts_async(snapshot_path, person_count, threat)

        # Set global unknown cooldown to prevent repeat alerts
        self.unknown_paused_until = datetime.now() + timedelta(seconds=self.config.unknown_cooldown_seconds)
        self.logger.info(f"Unknown alert cooldown: {self.config.unknown_cooldown_seconds}s")

    def handle_person_no_face(self, frame: np.ndarray, track_ids: List[int]):
        """Person detected but face voting timed out or no face visible"""
        self.logger.info(f"Person detected, face not identifiable (tracks: {track_ids})")

        # Start video recording FIRST (see handle_unknown_person)
        if self.video_recorder is not None:
            self.video_recorder.start_recording('no_face')

        person_count = len(track_ids)
        snapshot_path = self.snapshot_manager.save_snapshot(frame, person_count, is_known=False)

        # Run AI threat analysis
        threat = self.threat_analyzer.analyze_threat(snapshot_path)

        # Treat as unknown — email + Discord fire in the background
        self._send_alerts_async(snapshot_path, person_count, threat)

        # Set unknown cooldown
        self.unknown_paused_until = datetime.now() + timedelta(seconds=self.config.unknown_cooldown_seconds)
        self.logger.info(f"Unknown alert cooldown: {self.config.unknown_cooldown_seconds}s")

    def _tamper_cooldown_active(self) -> bool:
        """Return True if a tamper alert was sent recently."""
        if self._last_tamper_alert is None:
            return False
        elapsed = (datetime.now() - self._last_tamper_alert).total_seconds()
        return elapsed < self.config.tamper_cooldown_seconds

    def handle_tamper_alert(self, tamper_type: str, snapshot_path: Optional[Path] = None):
        """Handle a tamper event (disconnect or obstruction).

        Creates a level-10 ThreatAssessment and sends email + Discord alerts.
        """
        self.logger.warning(f"TAMPER ALERT: {tamper_type} detected")
        self._last_tamper_alert = datetime.now()

        threat = ThreatAssessment(
            threat_level=10,
            description=f"Camera {tamper_type} - potential tampering",
            reasoning=(
                f"Camera {tamper_type} detected while sentry was active. "
                "Treating as unknown threat."
            ),
        )

        # Start video recording FIRST for obstruction events (camera still works)
        if tamper_type == 'obstruction' and self.video_recorder is not None:
            self.video_recorder.start_recording('tamper')

        if snapshot_path is not None:
            self._send_alerts_async(snapshot_path, 0, threat)
        else:
            threading.Thread(
                target=self._send_tamper_alert_no_image,
                args=(tamper_type, threat), daemon=True
            ).start()

    def _send_tamper_alert_no_image(self, tamper_type: str, threat: ThreatAssessment):
        """Send tamper email + Discord without a snapshot (camera is dead)."""
        est_time = datetime.now(ZoneInfo("America/New_York"))
        timestamp = est_time.strftime("%m/%d/%Y %I:%M:%S %p EST")

        # --- Email ---
        if self.config.email_sender and self.config.email_app_password:
            try:
                subject = f"[CRITICAL] MUSTANG TAMPER ALERT: Camera {tamper_type} - {timestamp}"
                email_body = (
                    f"VIVIAN Sentry System - TAMPER ALERT\n\n"
                    f"THREAT ASSESSMENT:\n"
                    f"{'=' * 40}\n"
                    f"Level: 10/10 (CRITICAL)\n"
                    f"Type: Camera {tamper_type}\n\n"
                    f"Analysis: {threat.reasoning}\n"
                    f"{'=' * 40}\n\n"
                    f"Time: {timestamp}\n"
                    f"Location: 1969 Mustang Interior (Dashboard Camera)\n\n"
                    f"No snapshot available - camera was not producing frames."
                )
                msg = EmailMessage()
                msg["From"] = self.config.email_sender
                msg["To"] = self.config.email_recipient
                msg["Subject"] = subject
                msg.set_content(email_body)

                with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                    server.starttls()
                    server.login(self.config.email_sender, self.config.email_app_password)
                    server.send_message(msg)
                self.logger.info(f"Tamper alert email sent to {self.config.email_recipient}")
            except Exception as e:
                self.logger.error(f"Failed to send tamper alert email: {e}")

        # --- Discord ---
        if self.config.discord_webhook_url:
            try:
                embed = {
                    "title": f"TAMPER ALERT - Camera {tamper_type}",
                    "description": (
                        f"**Level:** 10/10 (CRITICAL)\n"
                        f"**Type:** Camera {tamper_type}\n"
                        f"**Analysis:** {threat.reasoning}"
                    ),
                    "color": 0x8B0000,  # Dark red
                    "fields": [
                        {"name": "Time", "value": timestamp, "inline": True},
                        {"name": "Location", "value": "1969 Mustang Interior", "inline": True},
                    ],
                    "footer": {"text": "VIVIAN Sentry System"},
                }
                resp = requests.post(
                    self.config.discord_webhook_url,
                    json={"embeds": [embed]},
                    timeout=15,
                )
                if resp.status_code in (200, 204):
                    self.logger.info("Tamper Discord alert sent successfully")
                else:
                    self.logger.error(f"Discord webhook returned {resp.status_code}: {resp.text}")
            except Exception as e:
                self.logger.error(f"Failed to send tamper Discord alert: {e}")

    def _send_alert_email(self, snapshot_path: Path, person_count: int = 1,
                          threat: Optional[ThreatAssessment] = None):
        if not self.config.email_sender or not self.config.email_app_password:
            self.logger.warning("Email credentials not configured - skipping email alert")
            return

        try:
            est_time = datetime.now(ZoneInfo("America/New_York"))
            timestamp = est_time.strftime("%m/%d/%Y %I:%M:%S %p EST")

            if threat and not threat.error:
                if self.config.threat_analysis_diagnostic_mode:
                    subject = f"[DIAGNOSTIC] MUSTANG SENTRY TEST - {timestamp}"
                    email_body = (
                        f"VIVIAN Sentry System - AI Diagnostic Mode\n\n"
                        f"DIAGNOSTIC TEST RESULT:\n"
                        f"{'=' * 40}\n"
                        f"Status: AI Analysis Working Correctly\n"
                        f"Mode: Diagnostic Test\n\n"
                        f"What the AI sees:\n"
                        f"{threat.reasoning}\n"
                        f"{'=' * 40}\n\n"
                        f"Time: {timestamp}\n"
                        f"Location: 1969 Mustang Interior (Dashboard Camera)\n\n"
                        f"NOTE: This is a DIAGNOSTIC test. Threat assessment is disabled.\n"
                        f"Photo attached below."
                    )
                else:
                    level = threat.threat_level
                    severity = "LOW" if level <= 3 else "MODERATE" if level <= 6 else "HIGH" if level <= 8 else "CRITICAL"
                    indicator = "[LOW]" if level <= 3 else "[MODERATE]" if level <= 6 else "[HIGH]" if level <= 8 else "[CRITICAL]"
                    subject = f"{indicator} MUSTANG ALERT: THREAT LEVEL {level}/10 - {timestamp}"
                    email_body = (
                        f"VIVIAN Sentry System - 1969 Mustang Interior Alert\n\n"
                        f"THREAT ASSESSMENT:\n"
                        f"{'=' * 40}\n"
                        f"Level: {level}/10 ({severity})\n"
                        f"Status: {threat.description}\n\n"
                        f"Analysis: {threat.reasoning}\n"
                        f"{'=' * 40}\n\n"
                        f"Persons detected: {person_count}\n"
                        f"Time: {timestamp}\n"
                        f"Location: 1969 Mustang Interior (Dashboard Camera)\n\n"
                        f"Photo attached below."
                    )
            else:
                subject = f"MUSTANG SECURITY EVENT: {timestamp}"
                email_body = (
                    f"VIVIAN Sentry System detected {person_count} unknown person(s).\n\n"
                    f"Time: {timestamp}\n"
                    f"Location: Vehicle Interior (Dashboard Camera)\n\n"
                    f"Threat analysis unavailable.\n"
                    f"Photo attached."
                )

            msg = EmailMessage()
            msg["From"] = self.config.email_sender
            msg["To"] = self.config.email_recipient
            msg["Subject"] = subject
            msg.set_content(email_body)

            if snapshot_path.exists():
                with open(snapshot_path, "rb") as f:
                    image_data = f.read()
                msg.add_attachment(
                    image_data,
                    maintype="image",
                    subtype="jpeg",
                    filename=snapshot_path.name
                )

            with smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port) as server:
                server.starttls()
                server.login(self.config.email_sender, self.config.email_app_password)
                server.send_message(msg)

            self.logger.info(f"Alert email sent to {self.config.email_recipient}")

        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")

    def _send_discord_alert(self, snapshot_path: Path, person_count: int = 1,
                            threat: Optional[ThreatAssessment] = None):
        if not self.config.discord_webhook_url:
            self.logger.warning("Discord webhook URL not configured - skipping Discord alert")
            return

        try:
            est_time = datetime.now(ZoneInfo("America/New_York"))
            timestamp = est_time.strftime("%m/%d/%Y %I:%M:%S %p EST")

            if threat and not threat.error:
                if self.config.threat_analysis_diagnostic_mode:
                    title = "DIAGNOSTIC - Sentry Test"
                    color = 0x3498DB  # Blue
                    description = (
                        f"**Mode:** Diagnostic Test\n"
                        f"**AI sees:** {threat.reasoning}"
                    )
                else:
                    level = threat.threat_level
                    if level <= 3:
                        color = 0x2ECC71  # Green
                        severity = "LOW"
                    elif level <= 6:
                        color = 0xF39C12  # Orange
                        severity = "MODERATE"
                    elif level <= 8:
                        color = 0xE74C3C  # Red
                        severity = "HIGH"
                    else:
                        color = 0x8B0000  # Dark red
                        severity = "CRITICAL"

                    title = f"MUSTANG ALERT - Threat Level {level}/10 ({severity})"
                    description = (
                        f"**Status:** {threat.description}\n"
                        f"**Analysis:** {threat.reasoning}"
                    )
            else:
                title = "MUSTANG SECURITY EVENT"
                color = 0xF39C12  # Orange
                description = "Threat analysis unavailable."

            embed = {
                "title": title,
                "description": description,
                "color": color,
                "fields": [
                    {"name": "Persons Detected", "value": str(person_count), "inline": True},
                    {"name": "Time", "value": timestamp, "inline": True},
                    {"name": "Location", "value": "1969 Mustang Interior", "inline": True},
                ],
                "footer": {"text": "VIVIAN Sentry System"},
            }

            # Attach snapshot as image in embed
            if snapshot_path.exists():
                embed["image"] = {"url": f"attachment://{snapshot_path.name}"}
                with open(snapshot_path, "rb") as f:
                    files = {"file": (snapshot_path.name, f, "image/jpeg")}
                    payload = {"payload_json": json.dumps({"embeds": [embed]})}
                    resp = requests.post(self.config.discord_webhook_url, data=payload, files=files, timeout=15)
            else:
                resp = requests.post(
                    self.config.discord_webhook_url,
                    json={"embeds": [embed]},
                    timeout=15,
                )

            if resp.status_code in (200, 204):
                self.logger.info("Discord alert sent successfully")
            else:
                self.logger.error(f"Discord webhook returned {resp.status_code}: {resp.text}")

        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {e}")

    def process_frame(self, frame: np.ndarray) -> None:
        """
        Process a single frame through the detection pipeline.

        Confirmation and face voting run in PARALLEL:
        - Every frame with a person feeds both the confirmation buffer AND face voting
        - Action is taken only when BOTH confirmation is met AND voting resolves
        """
        # Frame-skip check FIRST — brightness/rotation are not free, and
        # running them on frames we then discard defeats the skip setting.
        self.frame_count += 1
        if self.frame_count % self.config.process_every_n_frames != 0:
            return

        frame = self.adjust_brightness(frame)

        if self.config.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Check if paused (known person cooldown)
        if self.paused_until is not None:
            if datetime.now() < self.paused_until:
                return
            else:
                self.logger.info("Known-person cooldown expired, resuming monitoring")
                self.paused_until = None

        # Check unknown person cooldown
        if self.unknown_paused_until is not None:
            if datetime.now() < self.unknown_paused_until:
                return
            else:
                self.logger.info("Unknown-person cooldown expired, resuming monitoring")
                self.unknown_paused_until = None
                # Reset detection state for fresh start
                self.person_detector.reset_confirmation()
                self._confirmed_no_face_frames = 0
                if self.face_recognizer:
                    self.face_recognizer.reset_voting()

        # Motion detection pre-filter
        if not self.motion_detector.detect(frame):
            if self.frame_count % 100 == 0:
                self.logger.info(f"[frame {self.frame_count}] No motion detected")
            return

        # Person detection (YOLO + ByteTrack)
        detections, annotated_frame = self.person_detector.detect(frame)

        if not detections:
            self.person_detector.confirm_detection([])
            if self.face_recognizer:
                self.face_recognizer.reset_voting()
            return

        # Feed confirmation buffer (runs every frame)
        confirmed = self.person_detector.confirm_detection(detections)

        # Feed face voting buffer IN PARALLEL with confirmation
        face_result = None
        if self.face_recognizer is not None:
            face_result = self.face_recognizer.identify_all_persons(frame, detections)

        # Log progress
        hits = sum(self.person_detector.confirmation_buffer)
        total = len(self.person_detector.confirmation_buffer)
        if face_result is None and self.face_recognizer:
            voting_status = {tid: len(buf) for tid, buf in self.face_recognizer.track_voting.items()}
            self.logger.info(
                f"[frame {self.frame_count}] {len(detections)}p | "
                f"confirm {hits}/{total} ({'OK' if confirmed else 'wait'}) | "
                f"votes {voting_status}"
            )
        elif face_result is not None:
            self.logger.info(
                f"[frame {self.frame_count}] {len(detections)}p | "
                f"confirm {'OK' if confirmed else f'{hits}/{total}'} | "
                f"voting RESOLVED"
            )

        # Need BOTH confirmation AND face result to act
        if not confirmed:
            self._confirmed_no_face_frames = 0
            return

        if self.face_recognizer is not None:
            if face_result is None:
                # Confirmed but still collecting face votes
                # Check if we've been waiting too long with NO votes at all
                has_any_votes = bool(self.face_recognizer.track_voting)
                if not has_any_votes:
                    self._confirmed_no_face_frames += 1
                    if self._confirmed_no_face_frames >= self.config.face_voting_timeout_frames:
                        # No face detected after timeout - treat as unidentifiable
                        self.logger.info(
                            f"No face detected after {self._confirmed_no_face_frames} confirmed frames - "
                            f"treating as unidentifiable person"
                        )
                        new_track_ids = self.person_detector.get_new_track_ids(detections)
                        if new_track_ids:
                            self.handle_person_no_face(annotated_frame, new_track_ids)
                        self._confirmed_no_face_frames = 0
                        self.person_detector.reset_confirmation()
                        return
                else:
                    self._confirmed_no_face_frames = 0
                return

            # Both confirmed and voting resolved - check cooldown then act
            new_track_ids = self.person_detector.get_new_track_ids(detections)
            if not new_track_ids:
                self.logger.info("All tracks already alerted recently, skipping")
                return

            avg_confidence = np.mean([
                r['confidence'] for r in face_result['all_results'].values()
            ]) if face_result['all_results'] else 0.0

            self.logger.info(f"RESULT: known={face_result['known_names']}, unknown={face_result['unknown_track_ids']}")

            if face_result['has_known']:
                # An unknown track resolved in the SAME batch used to be
                # discarded entirely: the known-person branch ran, sentry stood
                # down (known_person_stops), and the stranger standing next to
                # the owner got no snapshot, no threat analysis and no alert.
                # Alert on them first, and only stand down once no unknown
                # tracks remain.
                unknown_tids = face_result['unknown_track_ids'] or []
                if unknown_tids:
                    self.logger.warning(
                        f"Known person present alongside unknown track(s) "
                        f"{unknown_tids} — alerting before standing down"
                    )
                    self.handle_unknown_person(annotated_frame, unknown_tids)

                self.handle_known_person(
                    face_result['known_names'],
                    avg_confidence,
                    annotated_frame
                )
                self.logger.log_detection(
                    len(detections),
                    np.mean([d['confidence'] for d in detections]),
                    ','.join(face_result['known_names']),
                    avg_confidence,
                    new_track_ids
                )
                if self.config.known_person_stops and not unknown_tids:
                    self.running = False
            else:
                unknown_tids = face_result['unknown_track_ids'] or new_track_ids
                self.handle_unknown_person(annotated_frame, unknown_tids)
                self.logger.log_detection(
                    len(detections),
                    np.mean([d['confidence'] for d in detections]),
                    'Unknown',
                    avg_confidence,
                    new_track_ids
                )

            # Timed-out tracks that were NOT already dispatched above. A
            # timed-out track lands in unknown_track_ids too, so without this
            # guard it ran the whole pipeline twice — two snapshots, two Claude
            # vision calls billed and serialized on this thread, two emails, two
            # Discord posts and two deterrents queued behind the TTS lock.
            for tid, res in face_result['all_results'].items():
                if res.get('timed_out') and tid in new_track_ids \
                        and tid not in unknown_tids:
                    self.handle_person_no_face(annotated_frame, [tid])

        else:
            # No face recognition - act on confirmation alone
            new_track_ids = self.person_detector.get_new_track_ids(detections)
            if not new_track_ids:
                return
            self.handle_unknown_person(annotated_frame, new_track_ids)
            self.logger.log_detection(
                len(detections),
                np.mean([d['confidence'] for d in detections]),
                'Degraded',
                0.0,
                new_track_ids
            )

    def periodic_maintenance(self, force: bool = False):
        """Hourly cleanup. Pass force=True at startup: the hourly gate meant a
        sentry run shorter than an hour never cleaned up at all, so disk use
        only ever grew on typical arm/disarm cycles."""
        now = datetime.now()
        if force or (now - self.last_cleanup).total_seconds() > 3600:
            self.snapshot_manager.cleanup_old_snapshots()
            self.person_detector.cleanup_old_tracks()
            self.last_cleanup = now

    def _setup_signal_handlers(self):
        """Register signal handlers for clean shutdown"""
        def handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            self.logger.info(f"Received {sig_name} - shutting down gracefully...")
            self.running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    def run(self):
        """Main detection loop"""
        self._setup_signal_handlers()
        self.logger.info("Starting Vehicle Sentry System Mk3 (with AI Threat Analysis)...")

        # Reclaim disk before arming, not an hour in — most runs are shorter
        # than the hourly maintenance gate, so cleanup never used to happen.
        try:
            self.periodic_maintenance(force=True)
        except Exception as e:
            self.logger.error(f"Startup maintenance failed: {e}")

        if self.watchdog.initialize_camera() is None:
            self.logger.error("Failed to initialize camera - exiting")
            return

        # Camera warm-up
        self.logger.info("Camera warming up (2 seconds)...")
        warmup_frames = 0
        warmup_start = time.time()
        while time.time() - warmup_start < 2.0:
            ret, _ = self.watchdog.read_frame()
            if ret:
                warmup_frames += 1
            time.sleep(0.05)
        self.logger.info(f"Warm-up complete ({warmup_frames} frames)")

        self.running = True
        self.motion_detector.start_time = time.time()
        self.logger.info("Sentry system active. Monitoring for persons...")
        self.logger.info(f"Motion warmup: bypassing motion filter for {self.config.motion_warmup_seconds}s")

        # Throttle recorder feeding to the recording fps (camera runs ~30fps;
        # unthrottled feeding overflows the writer queue -> slow-motion clips)
        rec_interval = 1.0 / max(self.config.recording_fps, 1.0)
        last_rec_feed = 0.0

        try:
            while self.running:
                ret, frame = self.watchdog.read_frame()

                # Enforce the recording duration limit every iteration, not just
                # when a frame is fed — the paths below `continue` past
                # feed_frame(), which would latch a recording on forever.
                if self.video_recorder is not None:
                    self.video_recorder.poll()

                if not ret:
                    if not self.watchdog.check_health():
                        self.logger.warning("Camera unhealthy - Loss of camera feed detected")
                        # Tamper: alert immediately on camera loss (before recovery attempts)
                        if not self._tamper_cooldown_active():
                            self.handle_tamper_alert('disconnect')
                        # Now attempt recovery
                        if not self.watchdog.attempt_recovery():
                            self.logger.error("Camera recovery failed - waiting before retry")
                            time.sleep(self.config.watchdog_retry_delay)
                            continue
                        # Reset motion detector and tamper detector after camera recovery
                        self.motion_detector.reset()
                        self.tamper_detector.reset()
                    else:
                        time.sleep(0.01)
                        continue

                # Validate frame before processing
                if frame is None or frame.size == 0:
                    self.logger.warning("Empty frame received, skipping")
                    continue

                # Tamper: obstruction check every N frames
                if self.frame_count % self.config.tamper_obstruction_check_interval == 0:
                    if self.tamper_detector.check_frame(frame):
                        if not self._tamper_cooldown_active():
                            # Save the suspicious frame as evidence
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            tamper_dir = self.snapshot_manager.unknown_dir
                            snapshot_path = tamper_dir / f"tamper_obstruction_{timestamp}.jpg"
                            cv2.imwrite(str(snapshot_path), frame)
                            self.logger.info(f"Tamper snapshot saved: {snapshot_path}")
                            self.handle_tamper_alert('obstruction', snapshot_path)
                        self.tamper_detector.reset()

                # Feed frame to video recorder (captures during cooldown too;
                # throttled to recording fps)
                _now = time.time()
                if self.video_recorder is not None and _now - last_rec_feed >= rec_interval:
                    last_rec_feed = _now
                    rec_frame = frame
                    if self.config.rotate_180:
                        rec_frame = cv2.rotate(frame, cv2.ROTATE_180)
                    self.video_recorder.feed_frame(rec_frame)

                self.process_frame(frame)
                self.periodic_maintenance()
                time.sleep(0.01)

        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self.running = False
            if self.video_recorder is not None:
                self.video_recorder.release()
            self.watchdog.release()
            self.logger.info("Sentry system stopped")

    def stop(self):
        self.running = False
        if self.video_recorder is not None:
            self.video_recorder.release()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VIVIAN Sentry System")
    parser.add_argument('--snapshot', action='store_true',
                        help='Capture a single frame as sentry_snapshot.jpg and exit')
    args = parser.parse_args()

    config = SentryConfig.from_yaml(CONFIG_FILE)

    if args.snapshot:
        cap = cv2.VideoCapture(config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Error: failed to capture frame")
            sys.exit(1)
        if config.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        out_path = Path(__file__).parent / 'sentry_snapshot.jpg'
        cv2.imwrite(str(out_path), frame)
        print(f"Snapshot saved to {out_path}")
        sys.exit(0)

    sentry = VehicleSentry(config)
    sentry.run()
