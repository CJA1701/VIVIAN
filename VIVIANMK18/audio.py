# Save this as: vivian/audio.py

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import scipy.signal
import torch
import subprocess
import os
import queue
import tempfile
from pathlib import Path
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)

# Every blocking step below runs on the MAIN thread inside the wake callback,
# where both audio watchdogs are disarmed — so anything unbounded here makes
# VIVIAN deaf until the callback watchdog hard-exits. All three are bounded.
CHUNK_TIMEOUT_S = 2.0        # a live mic delivers a 0.25s chunk every 0.25s;
                             # nothing arriving for 2s means the capture wedged
WHISPER_TIMEOUT_S = 60.0     # base.en on a Pi 5 does the 15s max clip in ~2s
# How long to wait for the user to START talking. Previously derived from
# silence_duration (0.7s * 3 = 1.5s), which was far too short: after the wake
# beep, LED change and mixer work, a driver who pauses to collect their thought
# got "No speech detected" and silence back — the classic "she ignored me".
INITIAL_SPEECH_TIMEOUT_S = 6.0


class AudioRecorder:
    """Handle audio recording with voice activity detection"""

    def __init__(self, config):
        self.config = config
        self.device_sample_rate = config.audio['sample_rate']  # Device native rate (48000)
        self.vad_sample_rate = 16000  # VAD and Whisper expect 16000Hz
        self.silence_duration = config.audio['silence_duration']
        self.max_duration = config.audio['max_record_duration']

        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        self.get_speech_timestamps = utils[0]
        logger.info("VAD model loaded")

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample audio from one sample rate to another"""
        if from_sr == to_sr:
            return audio

        # Calculate number of output samples
        num_samples = int(len(audio) * to_sr / from_sr)
        resampled = scipy.signal.resample(audio, num_samples)
        return resampled.astype(audio.dtype)

    def _find_mic_device(self) -> int:
        """Find the vivian_mic device index"""
        logger.info("Searching for vivian_mic device...")
        all_devices = sd.query_devices()

        for i, dev in enumerate(all_devices):
            if dev['max_input_channels'] > 0:
                logger.debug(f"  Input device {i}: {dev['name']} ({dev['max_input_channels']} channels)")
            if dev['max_input_channels'] > 0 and 'vivian_mic' in dev['name']:
                logger.info(f"Found vivian_mic device: {dev['name']} (index {i}, {dev['max_input_channels']} input channels)")
                return i

        logger.warning("vivian_mic device not found! Listing all input devices:")
        for i, dev in enumerate(all_devices):
            if dev['max_input_channels'] > 0:
                logger.warning(f"  Available input {i}: {dev['name']}")
        logger.warning("Falling back to default input device")
        return None

    def record(self) -> str:
        """Record audio until silence is detected, return path to WAV file"""
        logger.info("Recording started")

        # Chunk duration at device sample rate
        chunk_duration = 0.25  # seconds
        device_chunk_size = int(self.device_sample_rate * chunk_duration)
        vad_chunk_size = int(self.vad_sample_rate * chunk_duration)

        silence_limit = int(self.silence_duration / chunk_duration)
        initial_speech_limit = int(INITIAL_SPEECH_TIMEOUT_S / chunk_duration)
        silence_counter = 0
        speech_detected_ever = False

        recorded_chunks = []  # Store chunks at device sample rate
        vad_buffer = deque(maxlen=int(1.5 / chunk_duration))  # Rolling buffer for VAD
        # monotonic, not wall clock: this Pi has no RTC battery, so NTP steps the
        # clock at boot and would corrupt the max-duration math.
        start_time = time.monotonic()

        try:
            mic_device = self._find_mic_device()

            # Callback + queue rather than the blocking stream.read(): read()
            # has no timeout, so a wedged USB capture (the documented brownout
            # failure) hung recording forever — max_duration below could never
            # fire because it is only checked AFTER read() returns.
            audio_q: "queue.Queue" = queue.Queue()

            def _on_audio(indata, frames, time_info, status):
                if status:
                    logger.debug(f"Input stream status: {status}")
                audio_q.put_nowait(indata.copy())

            with sd.InputStream(
                device=mic_device,
                samplerate=self.device_sample_rate,
                channels=1,
                dtype="int16",
                blocksize=device_chunk_size,
                callback=_on_audio,
            ):
                logger.info(f"Recording at {self.device_sample_rate}Hz, resampling to {self.vad_sample_rate}Hz for VAD")

                while True:
                    try:
                        chunk = audio_q.get(timeout=CHUNK_TIMEOUT_S)
                    except queue.Empty:
                        raise RuntimeError(
                            f"Microphone delivered no audio for {CHUNK_TIMEOUT_S}s "
                            f"— capture stream wedged"
                        )

                    chunk_flat = chunk.flatten()
                    recorded_chunks.append(chunk_flat.copy())

                    # Log audio levels for debugging
                    max_amp = np.max(np.abs(chunk_flat))
                    if len(recorded_chunks) <= 4:  # Log first few chunks
                        logger.debug(f"Chunk {len(recorded_chunks)}: max amplitude = {max_amp}")

                    # Resample to 16000Hz for VAD
                    chunk_float = chunk_flat.astype(np.float32) / 32768.0
                    chunk_16k = self._resample(chunk_float, self.device_sample_rate, self.vad_sample_rate)
                    vad_buffer.append(chunk_16k)

                    # Check for speech in rolling buffer (at 16000Hz)
                    buffer_audio = np.concatenate(list(vad_buffer)) if vad_buffer else np.array([], dtype=np.float32)
                    audio_tensor = torch.from_numpy(buffer_audio).unsqueeze(0)

                    speech_segments = self.get_speech_timestamps(
                        audio_tensor,
                        self.vad_model,
                        sampling_rate=self.vad_sample_rate  # Tell VAD we're at 16000Hz
                    )

                    # Check if speech detected in last 0.5 seconds of buffer
                    samples_for_half_sec = int(0.5 * self.vad_sample_rate)  # At 16000Hz
                    speaking = any(
                        seg["end"] >= len(buffer_audio) - samples_for_half_sec
                        for seg in speech_segments
                    )

                    if speaking:
                        speech_detected_ever = True
                        silence_counter = 0
                        logger.debug("Speech detected")
                    else:
                        silence_counter += 1

                    # Only stop on silence if we've detected speech before
                    # This prevents stopping immediately on startup
                    if speech_detected_ever and silence_counter >= silence_limit:
                        logger.info(f"Silence detected after speech, stopping recording ({len(recorded_chunks)} chunks)")
                        break

                    # Also stop if the user never started talking. Uses its own
                    # generous timeout — NOT a multiple of silence_duration,
                    # which only governs how long a trailing pause ends a
                    # sentence the user already started.
                    if not speech_detected_ever and silence_counter >= initial_speech_limit:
                        logger.warning("No speech detected, stopping recording")
                        break

                    if time.monotonic() - start_time > self.max_duration:
                        logger.info("Max duration reached")
                        break

        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
            raise

        # Combine all recorded chunks
        if recorded_chunks:
            full_recording = np.concatenate(recorded_chunks)
            logger.info(f"Total samples recorded: {len(full_recording)} at {self.device_sample_rate}Hz")
            logger.info(f"Max amplitude in recording: {np.max(np.abs(full_recording))}")

            # Resample to 16000Hz for Whisper
            full_recording_float = full_recording.astype(np.float32) / 32768.0
            full_recording_16k = self._resample(full_recording_float, self.device_sample_rate, self.vad_sample_rate)

            # Apply software gain boost to improve transcription accuracy
            # The USB mic input is quite weak even at max hardware gain
            # Note: Too much gain causes clipping/distortion - keep moderate
            gain_boost = 2.0
            full_recording_16k_boosted = full_recording_16k * gain_boost
            # Clip to prevent distortion
            full_recording_16k_boosted = np.clip(full_recording_16k_boosted, -1.0, 1.0)
            logger.info(f"Applied {gain_boost}x software gain boost")

            full_recording_16k_int = (full_recording_16k_boosted * 32767).astype(np.int16)
        else:
            full_recording_16k_int = np.array([], dtype=np.int16)

        # Save recording to temporary file at 16000Hz (what Whisper expects)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        wavfile.write(tmp_path, self.vad_sample_rate, full_recording_16k_int)
        logger.info(f"Recording saved to {tmp_path} at {self.vad_sample_rate}Hz")
        return tmp_path


class AudioTranscriber:
    """Transcribe audio using whisper.cpp"""

    def __init__(self, config):
        self.config = config
        self.whisper_binary = config.whisper['binary_path']
        self.model_path = config.whisper['model_path']

        # Verify files exist
        if not os.path.isfile(self.whisper_binary):
            raise FileNotFoundError(f"Whisper binary not found: {self.whisper_binary}")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Whisper model not found: {self.model_path}")

        logger.info("Audio transcriber initialized")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Deletes the temp recording and whisper transcript afterwards —
        /tmp is tmpfs (RAM) on the Pi, so leaked files add up over weeks.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        command = [
            self.whisper_binary,
            "-m", self.model_path,
            "-otxt",
            "-l", "en",
            audio_path
        ]
        transcript_file = audio_path + ".txt"

        try:
            try:
                # Bounded: this runs on the main thread inside the wake callback,
                # so a wedged whisper-cli (e.g. stalled on the mmapped model
                # after an SD-card hiccup) would otherwise block VIVIAN until
                # the callback watchdog hard-exits. base.en handles the 15s max
                # recording in a couple of seconds on a Pi 5, so 60s is generous.
                subprocess.run(command, check=True, capture_output=True,
                               timeout=WHISPER_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.error(f"Whisper transcription timed out after {WHISPER_TIMEOUT_S}s")
                raise RuntimeError("Transcription timed out")
            except subprocess.CalledProcessError as e:
                logger.error(f"Whisper transcription failed: {e}")
                raise RuntimeError(f"Transcription failed: {e}")

            if not os.path.isfile(transcript_file):
                raise FileNotFoundError(f"Transcript not created: {transcript_file}")

            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            logger.info(f"Transcribed: {text}")
            return text
        finally:
            for path in (transcript_file, audio_path):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass
