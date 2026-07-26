# VIVIAN Mk18 TTS — local synthesis with per-voice effect chains.
#
# Two engines share one public surface (speak / stream_speaker /
# synthesize_to_wav / set_output_device / OUTPUT_* constants) so callers are
# unchanged:
#   * PiperTTS  (default) — native Piper binary, one ONNX model per voice.
#   * KokoroTTS (fallback) — kokoro-onnx, kept for rollback.
#
# Voice PROFILES let different callers use different voices from the same
# engine instance:
#   * "default" — the assistant voice (alba + a light synthetic chain).
#   * "sentry"  — GLaDOS, used by sentry mode's threat/deterrent lines.
#
# Pipeline (per sentence): text -> engine synth (mono WAV) -> ffmpeg effect
# chain + resample to S16LE 48kHz stereo -> aplay, while the next sentence
# synthesizes in the background. Piper runs at RTF ~0.33 on the Pi 5, so
# synthesis stays well ahead of playback and the stream never underruns.

import json
import logging
import math
import os
import queue
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Shared file for CRT waveform display
TTS_AMPLITUDE_FILE = "/tmp/vivian_tts_amplitude.json"

# Output device options - easily switchable for different modules
# Note: stereo_direct bypasses the mute control so VIVIAN can speak while Spotify is muted
OUTPUT_STEREO = "stereo_direct"    # External/car stereo (bypasses mute)
OUTPUT_INTERNAL = "internal_plug"  # Internal speakers (for Sentry module)

PLAYBACK_RATE = 48000  # what the ALSA devices expect (S16LE stereo)

# Upper bound on how long _StreamSpeaker.close() will wait for queued audio to
# finish. Generous — a long streamed reply can take a while to speak — but it
# MUST be bounded: close() runs inside the wake callback, so hanging here means
# VIVIAN never returns to listening for the wake word.
STREAM_CLOSE_TIMEOUT_S = 180.0

DEFAULT_PROFILE = "default"

# Effect chains, expressed as the *middle* of an ffmpeg filtergraph (the base
# class prepends "[0:a]", appends the gain stage + "[out]", and invokes it via
# -filter_complex). A chain may be a simple comma chain or a multi-pad graph
# (labels + ';'). None = no effect (engine output is only resampled).
EFFECT_CHAINS = {
    "none": None,

    # Legacy Kokoro "glados" chain (pitch up ~7% + dual vibrato + chorus).
    # Kept for the kokoro fallback engine.
    "glados_kokoro": (
        "asetrate=25680,aresample=24000,atempo=0.93458,"
        "vibrato=f=4:d=0.3,vibrato=f=0.55:d=0.2,"
        "chorus=0.5:0.9:45:0.35:0.25:1.8"
    ),

    # Assistant voice colour for the alba Piper voice: eased chorus doubling
    # + light echo, with a subtle (11%) ring-mod layer blended under the dry
    # signal for a faint synthetic edge (no pitch sway). Tuned with the user.
    "alba": (
        "chorus=0.6:0.9:50:0.3:0.2:0.8,aecho=0.8:0.8:30:0.2,asplit=2[d][w];"
        "[w]aeval=val(0)*sin(2*PI*t*55)|val(1)*sin(2*PI*t*55),volume=0.11[rm];"
        "[d][rm]amix=inputs=2:normalize=0,alimiter=limit=0.97"
    ),
}


class _VoiceProfile:
    """One named voice: an engine-specific source (Piper model / Kokoro voice)
    plus an effect chain and gain."""

    __slots__ = ("name", "effect", "gain_db", "voice", "model")

    def __init__(self, name, effect, gain_db, voice=None, model=None):
        self.name = name
        self.effect = effect
        self.gain_db = gain_db
        self.voice = voice   # Kokoro voice id (e.g. af_alloy)
        self.model = model   # Piper .onnx path

    def __repr__(self):
        src = f"model={self.model}" if self.model else f"voice={self.voice}"
        return f"<VoiceProfile {self.name} {src} effect={self.effect} gain={self.gain_db}>"


def TextToSpeech(config, output_device=OUTPUT_STEREO):
    """Factory: build the TTS engine named by config.tts.engine.

    Callers keep doing `from tts import TextToSpeech; tts = TextToSpeech(cfg)`.
    """
    engine = (config.tts.get("engine") or "piper").lower()
    if engine == "kokoro":
        return KokoroTTS(config, output_device)
    if engine != "piper":
        logger.warning(f"Unknown tts.engine '{engine}', defaulting to piper")
    return PiperTTS(config, output_device)


class _BaseTTS:
    """Shared machinery: profiles, ffmpeg effect stage, chunked playback,
    the CRT amplitude contract, and the streaming sink.

    Subclasses implement _synth_raw_wav(text, profile) -> path to a mono WAV
    in the voice's native sample rate.
    """

    # engine 'kind' controls legacy-profile defaults; overridden by subclass
    _kind = "base"

    def __init__(self, config, output_device=OUTPUT_STEREO):
        self.config = config
        self.output_device = output_device
        self.base = Path(__file__).parent
        tts_cfg = config.tts

        self.speed = float(tts_cfg.get("speed", 1.0))
        self.profiles = self._build_profiles(tts_cfg)
        # Pre-build the ffmpeg filtergraph for each profile.
        self._graphs = {name: self._filtergraph_for(p)
                        for name, p in self.profiles.items()}

        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg is required for TTS but was not found on PATH")

        # Synthesis may be called from the assistant and sentry threads at
        # once; subclasses that aren't process-isolated use this lock.
        self._synth_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    def _build_profiles(self, tts_cfg) -> dict:
        default_gain = float(tts_cfg.get("gain_db", 6.0))
        raw = tts_cfg.get("profiles")
        profiles: dict = {}
        if raw:
            for name, spec in raw.items():
                spec = spec or {}
                profiles[name] = _VoiceProfile(
                    name=name,
                    effect=spec.get("effect", "none"),
                    gain_db=float(spec.get("gain_db", default_gain)),
                    voice=spec.get("voice"),
                    model=spec.get("model"),
                )
        else:
            # Legacy single-voice config (top-level voice/effect keys).
            legacy_effect = tts_cfg.get(
                "effect", "glados_kokoro" if self._kind == "kokoro" else "none"
            )
            profiles[DEFAULT_PROFILE] = _VoiceProfile(
                name=DEFAULT_PROFILE,
                effect=legacy_effect,
                gain_db=default_gain,
                voice=tts_cfg.get("voice", "af_alloy"),
                model=tts_cfg.get("model"),
            )
        if DEFAULT_PROFILE not in profiles:
            # Alias the first declared profile as the default.
            first = next(iter(profiles.values()))
            profiles[DEFAULT_PROFILE] = first
        return profiles

    def _profile(self, name) -> _VoiceProfile:
        return self.profiles.get(name) or self.profiles[DEFAULT_PROFILE]

    def _filtergraph_for(self, profile: _VoiceProfile):
        """Build the full -filter_complex graph string for a profile, or None
        if the audio only needs resampling (no effect, no gain)."""
        effect = profile.effect
        if effect not in EFFECT_CHAINS:
            logger.warning(f"Unknown TTS effect '{effect}' for profile "
                           f"'{profile.name}', treating as none")
            effect = "none"
        body = EFFECT_CHAINS[effect]

        gain = None
        if profile.gain_db:
            gain = f"volume={profile.gain_db}dB,alimiter=limit=0.97"

        if not body and not gain:
            return None
        if body and gain:
            inner = f"{body},{gain}"
        else:
            inner = body or gain
        return f"[0:a]{inner}[out]"

    # ------------------------------------------------------------------
    # Warm-up (throwaway synth so the first real response isn't slow)
    # ------------------------------------------------------------------

    def _warmup(self):
        for name in dict.fromkeys([DEFAULT_PROFILE, *self.profiles.keys()]):
            try:
                pcm = self._synthesize_pcm("ready", name)
                self._cleanup(pcm)
            except Exception as e:
                logger.debug(f"TTS warm-up skipped for profile '{name}': {e}")

    def set_output_device(self, device: str):
        """Switch output device (use OUTPUT_STEREO or OUTPUT_INTERNAL)"""
        self.output_device = device
        logger.info(f"TTS output device changed to: {device}")

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _synth_raw_wav(self, text: str, profile: _VoiceProfile) -> str:
        """Synthesize text in the profile's voice to a mono WAV file (native
        sample rate). Return the path; caller owns/deletes it."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Effect stage + PCM production
    # ------------------------------------------------------------------

    def _synthesize_pcm(self, text: str, profile_name: str = DEFAULT_PROFILE) -> str:
        """Synthesize text and return path to an S16LE 48kHz stereo PCM file
        with the profile's effect chain applied. Caller owns the file."""
        profile = self._profile(profile_name)
        wav_path = self._synth_raw_wav(text, profile)

        pcm_fd, pcm_path = tempfile.mkstemp(suffix=".raw")
        os.close(pcm_fd)
        graph = self._graphs.get(profile.name)
        try:
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", wav_path]
            if graph:
                cmd += ["-filter_complex", graph, "-map", "[out]"]
            cmd += ["-f", "s16le", "-ac", "2", "-ar", str(PLAYBACK_RATE), pcm_path]
            subprocess.run(cmd, timeout=60, check=True)
            return pcm_path
        except Exception:
            self._cleanup(pcm_path)
            raise
        finally:
            self._cleanup(wav_path)

    # ------------------------------------------------------------------
    # Speaking (chunked pipeline)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list:
        """Split into sentence chunks; merge fragments so chunks aren't choppy."""
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
        if not parts:
            return []
        chunks = []
        for part in parts:
            if chunks and len(part) < 20:
                chunks[-1] = chunks[-1] + " " + part
            else:
                chunks.append(part)
        return chunks

    def speak(self, text: str, profile: str = DEFAULT_PROFILE) -> None:
        """Synthesize and play text. Blocks until playback finishes.

        Sentences are synthesized in a background thread while earlier ones
        play, so first audio starts after ~one sentence of synthesis.
        """
        if not text or not text.strip():
            return
        logger.info(f"Speaking ({profile}): {text[:50]}...")

        chunks = self._split_sentences(text)
        pcm_queue: "queue.Queue" = queue.Queue(maxsize=2)

        def producer():
            for chunk in chunks:
                try:
                    pcm_queue.put(self._synthesize_pcm(chunk, profile))
                except Exception as e:
                    logger.error(f"TTS synthesis failed for chunk '{chunk[:40]}': {e}")
            pcm_queue.put(None)  # end of speech

        threading.Thread(target=producer, daemon=True).start()

        try:
            while True:
                pcm_path = pcm_queue.get()
                if pcm_path is None:
                    break
                try:
                    self._play_pcm(pcm_path)
                finally:
                    self._cleanup(pcm_path)
            logger.info("Playback complete")
        finally:
            self._write_amplitude(0.0)

    def stream_speaker(self, profile: str = DEFAULT_PROFILE) -> "_StreamSpeaker":
        """Return a sink for speaking sentences as they arrive (streaming
        assistant responses). Feed sentences with .feed(), then .close()."""
        return _StreamSpeaker(self, profile)

    def _play_pcm(self, pcm_path: str) -> None:
        """Play one PCM chunk via aplay while tracking amplitude for the CRT."""
        start_event = threading.Event()
        tracker = threading.Thread(
            target=self._amplitude_tracker, args=(pcm_path, start_event), daemon=True
        )
        tracker.start()
        start_event.set()

        result = subprocess.run(
            ["aplay", "-q", "-D", self.output_device, "-f", "S16_LE",
             "-c", "2", "-r", str(PLAYBACK_RATE), pcm_path],
            stderr=subprocess.PIPE, timeout=120,
        )
        if result.returncode != 0 and result.stderr:
            logger.warning(f"aplay stderr: {result.stderr.decode().strip()}")
        tracker.join(timeout=2)

    # ------------------------------------------------------------------
    # Pre-caching (sentry lines, welcome WAVs)
    # ------------------------------------------------------------------

    def synthesize_to_wav(self, text: str, out_path, apply_fx: bool = True,
                          profile: str = DEFAULT_PROFILE) -> bool:
        """Render text to a 48kHz stereo WAV file (for pre-cached lines).

        Returns True on success.
        """
        prof = self._profile(profile)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            wav_path = self._synth_raw_wav(text, prof)
        except Exception as e:
            logger.error(f"synthesize_to_wav synth failed for '{text[:40]}': {e}")
            return False
        graph = self._graphs.get(prof.name) if apply_fx else None
        try:
            cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", wav_path]
            if graph:
                cmd += ["-filter_complex", graph, "-map", "[out]"]
            cmd += ["-f", "wav", "-ac", "2", "-ar", str(PLAYBACK_RATE), str(out_path)]
            subprocess.run(cmd, timeout=60, check=True)
            return True
        except Exception as e:
            logger.error(f"synthesize_to_wav ffmpeg failed for '{text[:40]}': {e}")
            return False
        finally:
            self._cleanup(wav_path)

    # ------------------------------------------------------------------
    # Amplitude tracking for CRT waveform (same contract as Mk17)
    # ------------------------------------------------------------------

    def _write_amplitude(self, amplitude: float):
        try:
            data = json.dumps({"amplitude": amplitude, "t": time.time()})
            tmp = TTS_AMPLITUDE_FILE + ".tmp"
            with open(tmp, 'w') as f:
                f.write(data)
            os.replace(tmp, TTS_AMPLITUDE_FILE)
        except Exception:
            pass

    def _compute_rms(self, pcm_bytes: bytes) -> float:
        if len(pcm_bytes) < 4:
            return 0.0
        n_samples = len(pcm_bytes) // 2
        try:
            samples = struct.unpack(f'<{n_samples}h', pcm_bytes[:n_samples * 2])
            sum_sq = sum(s * s for s in samples)
            rms = math.sqrt(sum_sq / n_samples) / 32768.0
            return min(1.0, rms * 5.4)  # Scale up for visibility
        except Exception:
            return 0.0

    def _amplitude_tracker(self, pcm_path, start_event):
        channels = 2
        bytes_per_sample = 2
        bytes_per_sec = PLAYBACK_RATE * channels * bytes_per_sample
        chunk_duration = 0.04  # 40ms chunks
        chunk_bytes = int(bytes_per_sec * chunk_duration)

        start_event.wait(timeout=10)
        start_time = time.time()

        try:
            with open(pcm_path, 'rb') as f:
                file_size = os.path.getsize(pcm_path)
                while True:
                    elapsed = time.time() - start_time
                    offset = int(elapsed * bytes_per_sec)
                    offset = offset - (offset % (channels * bytes_per_sample))

                    if offset >= file_size:
                        break

                    f.seek(offset)
                    chunk = f.read(chunk_bytes)
                    if not chunk:
                        break

                    amp = self._compute_rms(chunk)
                    self._write_amplitude(amp)
                    time.sleep(chunk_duration)
        except Exception:
            pass
        finally:
            self._write_amplitude(0.0)

    @staticmethod
    def _cleanup(path):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


class PiperTTS(_BaseTTS):
    """Native Piper (rhasspy) synthesis: one ONNX voice model per profile,
    spawned per sentence. No persistent model state, so profiles are just
    different model files and threads don't contend."""

    _kind = "piper"

    def __init__(self, config, output_device=OUTPUT_STEREO):
        super().__init__(config, output_device)
        tts_cfg = config.tts

        self.piper_bin = self._resolve(tts_cfg.get("piper_bin", "piper/piper"))
        if not os.access(self.piper_bin, os.X_OK):
            raise RuntimeError(f"Piper binary is not executable: {self.piper_bin}")
        self.piper_dir = self.piper_bin.parent

        # espeak-ng data ships beside the binary; let piper auto-locate it, but
        # make the bundled shared libs findable.
        self._env = os.environ.copy()
        lib = str(self.piper_dir)
        prev = self._env.get("LD_LIBRARY_PATH")
        self._env["LD_LIBRARY_PATH"] = f"{lib}:{prev}" if prev else lib

        # Resolve + validate each profile's model file.
        for name, prof in self.profiles.items():
            if not prof.model:
                raise ValueError(f"Piper profile '{name}' has no 'model' set")
            prof.model = self._resolve(prof.model)
            if not prof.model.exists():
                raise FileNotFoundError(
                    f"Piper voice model not found for profile '{name}': "
                    f"{prof.model} — run setup_mk18.sh to download voices"
                )

        # List each distinct profile once (a bare "default" may alias a named
        # profile object; dedupe by identity so the alias isn't logged twice).
        seen, parts = set(), []
        for name, prof in self.profiles.items():
            if id(prof) in seen:
                continue
            seen.add(id(prof))
            parts.append(f"{name}={prof.model.name} (fx={prof.effect})")
        logger.info(f"Piper TTS ready [{self.piper_bin.name}] profiles: "
                    f"{', '.join(parts)}, output={self.output_device}")

        self._warmup()

    def _resolve(self, path_str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (self.base / p)

    def _synth_raw_wav(self, text: str, profile: _VoiceProfile) -> str:
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        try:
            subprocess.run(
                [str(self.piper_bin), "-m", str(profile.model), "-f", wav_path],
                input=text.encode("utf-8"),
                env=self._env,
                cwd=str(self.piper_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=60,
                check=True,
            )
            return wav_path
        except Exception:
            self._cleanup(wav_path)
            raise


class KokoroTTS(_BaseTTS):
    """Fallback engine: kokoro-onnx. One shared model; profiles select the
    Kokoro voice id. Kept so the system can roll back to the Mk18-launch voice."""

    _kind = "kokoro"

    def __init__(self, config, output_device=OUTPUT_STEREO):
        super().__init__(config, output_device)
        tts_cfg = config.tts

        self.model_path = self._resolve(tts_cfg["model_path"])
        self.voices_path = self._resolve(tts_cfg["voices_path"])
        self.onnx_threads = int(tts_cfg.get("onnx_threads", 3))

        logger.info(f"Loading Kokoro TTS model ({self.model_path.name})...")
        start = time.time()
        import onnxruntime as ort
        from kokoro_onnx import Kokoro
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.onnx_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(
                str(self.model_path), sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self._kokoro = Kokoro.from_session(sess, str(self.voices_path))
        except Exception as e:
            logger.warning(f"Tuned ONNX session unavailable ({e}); using default")
            self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        logger.info(f"Kokoro loaded in {time.time() - start:.1f}s "
                    f"(threads={self.onnx_threads}, output={self.output_device})")

        self._warmup()

    def _resolve(self, path_str) -> Path:
        p = Path(path_str)
        p = p if p.is_absolute() else self.base / p
        if not p.exists():
            raise FileNotFoundError(
                f"Kokoro model file not found: {p} — run setup_mk18.sh to download it"
            )
        return p

    def _synth_raw_wav(self, text: str, profile: _VoiceProfile) -> str:
        import soundfile as sf
        voice = profile.voice or "af_alloy"
        with self._synth_lock:
            samples, sr = self._kokoro.create(
                text, voice=voice, speed=self.speed, lang="en-us"
            )
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        try:
            sf.write(wav_path, samples, sr)
            return wav_path
        except Exception:
            self._cleanup(wav_path)
            raise


class _StreamSpeaker:
    """Speaks sentences fed to it over time, in order, gaplessly.

    Two-stage pipeline (synth thread -> play thread) so the next sentence
    synthesizes while the current one plays. Used to voice a streaming
    assistant response as Claude produces it. Feed sentences with feed();
    close() blocks until playback of everything fed so far is finished.
    """

    def __init__(self, tts: _BaseTTS, profile: str = DEFAULT_PROFILE):
        self._tts = tts
        self._profile = profile
        self._sentence_q: "queue.Queue" = queue.Queue()
        self._pcm_q: "queue.Queue" = queue.Queue(maxsize=2)
        self._synth_thread = threading.Thread(target=self._synth_loop, daemon=True)
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def _synth_loop(self):
        # The sentinel must reach the play thread on EVERY exit path, or
        # _play_loop blocks forever on get() and close() never returns.
        try:
            while True:
                sentence = self._sentence_q.get()
                if sentence is None:
                    return
                try:
                    self._pcm_q.put(self._tts._synthesize_pcm(sentence, self._profile))
                except Exception as e:
                    logger.error(f"Stream TTS synthesis failed for '{sentence[:40]}': {e}")
        finally:
            self._pcm_q.put(None)

    def _play_loop(self):
        try:
            while True:
                pcm_path = self._pcm_q.get()
                if pcm_path is None:
                    break
                try:
                    self._tts._play_pcm(pcm_path)
                except Exception as e:
                    # One failed chunk (aplay error, playback timeout) must not
                    # kill this thread: that would truncate the rest of the
                    # reply and strand the synth thread on a full queue.
                    logger.error(f"Stream TTS playback failed: {e}")
                finally:
                    self._tts._cleanup(pcm_path)
        finally:
            self._tts._write_amplitude(0.0)

    def feed(self, sentence: str) -> None:
        if sentence and sentence.strip():
            self._sentence_q.put(sentence.strip())

    def close(self) -> None:
        """Signal end of speech and block until all fed audio has played.

        Bounded: an unbounded join here would hang the whole wake callback (and
        therefore VIVIAN's return to listening) if playback ever wedged.
        """
        self._sentence_q.put(None)
        self._play_thread.join(timeout=STREAM_CLOSE_TIMEOUT_S)
        if self._play_thread.is_alive():
            logger.error(
                f"Stream TTS did not finish within {STREAM_CLOSE_TIMEOUT_S:.0f}s "
                f"— abandoning playback to return to wake word listening"
            )
