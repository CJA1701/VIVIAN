# VIVIAN Mk18 assistant brain — Anthropic Claude Messages API with tool use.
#
# Replaces the Mk17 OpenAI Chat Completions implementation. Public surface
# is unchanged (Assistant(...), .process_request(), .sentry_requested) so
# main.py needs no modification.

import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

import anthropic
import pytz

logger = logging.getLogger(__name__)

# Tools that short-circuit the model loop: execute, speak a canned
# announcement (or nothing), and end the turn without a second API
# round-trip. Keeps music/system commands as fast as they were on Mk17.
FAST_PATH_TOOLS = {
    "control_music", "play_track", "play_album", "play_artist",
    "reboot_system", "shutdown_system", "activate_sentry_mode",
    "remember_fact",
}

MAX_TOOL_ITERATIONS = 6


class Assistant:
    """Claude-powered assistant with tool use and server-side web search"""

    def __init__(self, config, system_info, memory_manager, spotify_controller, tts,
                 sentry_controller=None):
        self.config = config
        self.system_info = system_info
        self.memory = memory_manager
        self.spotify = spotify_controller
        self.tts = tts
        self.sentry_controller = sentry_controller

        # Sentry activation flag - checked by main.py after interaction
        self.sentry_requested = False

        self.model = config.anthropic.get('model', 'claude-opus-4-8')
        # SDK retries 429/5xx/connection errors automatically (max_retries).
        # Short timeout: a dead cellular zone should fail the interaction in
        # seconds (main.py speaks an error), not hang the wake loop.
        self.client = anthropic.Anthropic(
            api_key=config.anthropic['api_key'],
            max_retries=2,
            timeout=20.0,
        )

        self.tools = self._define_tools()
        self.timezone = pytz.timezone(config.timezone)
        self._last_error = None

        # Durable facts VIVIAN remembers across drives (written by the
        # remember_fact tool, injected into the system prompt each turn).
        self.facts_file = Path(__file__).parent / "vivian_facts.txt"
        self.max_facts = 40

        logger.info(f"Assistant initialized on {self.model} with tool use and web search")

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _web_search_tool_type(self) -> str:
        """Pick the web-search tool version the configured model supports.

        The _20260209 'dynamic filtering' variant runs code execution under
        the hood and only works on Opus 4.6+/Sonnet 4.6+/Fable 5. Haiku (and
        older models) must use the basic _20250305 variant — otherwise the
        API 400s on every request.
        """
        m = self.model
        dynamic_ok = any(tag in m for tag in (
            "opus-4-8", "opus-4-7", "opus-4-6", "sonnet-5", "sonnet-4-6", "fable-5"
        ))
        return "web_search_20260209" if dynamic_ok else "web_search_20250305"

    def _define_tools(self) -> List[Dict]:
        """Define tools Claude can call. Order is stable for prompt caching."""
        return [
            # Server-side tool: Anthropic runs the search, results come back
            # with citations — replaces the old DuckDuckGo scraping. Version
            # is model-dependent (see _web_search_tool_type).
            {
                "type": self._web_search_tool_type(),
                "name": "web_search",
                "max_uses": 3,
            },
            {
                "name": "control_music",
                "description": (
                    "Control Spotify music playback. Use action 'play_playlist' "
                    "to start the user's default driving playlist (their usual/main "
                    "playlist) — call it directly whenever they ask for 'my playlist', "
                    "'my driving playlist', 'the usual', or just to put music on. Do NOT "
                    "ask which playlist; there is one default and this action plays it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["skip", "previous", "pause", "resume", "play_playlist"],
                            "description": (
                                "The music control action. 'play_playlist' starts the "
                                "user's default driving playlist from the beginning."
                            )
                        }
                    },
                    "required": ["action"]
                }
            },
            {
                "name": "play_track",
                "description": (
                    "Play a specific song on Spotify. ALWAYS call this with your "
                    "best interpretation of what the user asked for — never ask "
                    "them to clarify a song title. Spotify's search is fuzzy and "
                    "resolves partial, misheard or approximate titles; a wrong "
                    "guess is trivially corrected by the driver, whereas a "
                    "clarifying question while they are driving is useless. "
                    "The request reached you via speech-to-text, so titles are "
                    "often mangled — especially numbers, which arrive split or "
                    "hyphenated ('in the year 25-25' is the song 'In the Year "
                    "2525'). Join split digits and pass the plausible real title."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "track_name": {
                            "type": "string",
                            "description": (
                                "Name of the track, normalised to the most likely "
                                "real title (join split digits: '25-25' -> '2525')"
                            )
                        },
                        "artist_name": {
                            "type": "string",
                            "description": "Name of the artist (optional)"
                        }
                    },
                    "required": ["track_name"]
                }
            },
            {
                "name": "play_album",
                "description": "Play a specific album on Spotify",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "album_name": {
                            "type": "string",
                            "description": "Name of the album to play"
                        },
                        "artist_name": {
                            "type": "string",
                            "description": "Name of the artist"
                        }
                    },
                    "required": ["album_name", "artist_name"]
                }
            },
            {
                "name": "play_artist",
                "description": "Play an artist's most popular songs on Spotify",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "artist_name": {
                            "type": "string",
                            "description": "Name of the artist"
                        }
                    },
                    "required": ["artist_name"]
                }
            },
            {
                "name": "get_current_song",
                "description": "Get information about the currently playing song on Spotify",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_weather",
                "description": "Get current weather conditions or forecast for a location. Use this when user asks about weather, temperature, rain, snow, forecast, etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name (optional, defaults to user's location)"
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["current", "forecast", "temperature", "conditions"],
                            "description": "Type of weather information requested"
                        }
                    },
                    "required": ["query_type"]
                }
            },
            {
                "name": "reboot_system",
                "description": "Reboot the Raspberry Pi system",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "shutdown_system",
                "description": "Shut down the Raspberry Pi system immediately. Use when user says 'shut down', 'power off', 'turn off', 'shutdown', or indicates they are done driving / turning off the car.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_current_location",
                "description": "Get the current GPS location with detailed address including street number, street name, city, state, and zip code. Use when user asks 'where am I', 'what's my address', 'what street am I on', etc.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "find_nearby_places",
                "description": "Find nearby places of interest like gas stations, restaurants, hotels, ATMs, etc. Use when user asks 'where is the nearest...', 'find a...', 'is there a... nearby', etc. Use this for GENERAL categories only.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "place_category": {
                            "type": "string",
                            "enum": ["gas_station", "restaurant", "fast_food", "cafe", "hotel", "atm", "bank", "pharmacy", "hospital", "police", "parking", "charging_station"],
                            "description": "Type of place to search for"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5)"
                        }
                    },
                    "required": ["place_category"]
                }
            },
            {
                "name": "find_specific_business",
                "description": "Find a specific business by name (e.g., 'McDonald's', 'Walmart', 'Shell', 'Starbucks'). Use when user asks for a SPECIFIC business name, not a general category. Example: 'find McDonald's' or 'where's the nearest Wendy's'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "business_name": {
                            "type": "string",
                            "description": "Name of the specific business to find (e.g., 'McDonald's', 'Shell', 'Walmart')"
                        }
                    },
                    "required": ["business_name"]
                }
            },
            {
                "name": "run_diagnostics",
                "description": "Run live system and network diagnostics on the VIVIAN Pi. Use when the user asks about connectivity, network status, internet connection, who is connected to the AP, CPU temperature, memory, disk space, uptime, or any question about the Pi's current hardware or software state. Always call this — never guess system state from memory.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["network", "system", "full"],
                            "description": "network = connectivity, interfaces, AP clients; system = CPU temp, RAM, disk, uptime; full = both"
                        }
                    },
                    "required": ["type"]
                }
            },
            {
                "name": "activate_sentry_mode",
                "description": "Activate vehicle sentry/security mode (camera monitoring). MUST be called for any request to guard/watch the car or enter sentry mode — never respond with text instead. Phonetic/garbled matches count (see system prompt rule 3).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "remember_fact",
                "description": "Save a durable fact about the driver, the car, or their preferences to remember across future drives (e.g. 'CJ prefers album playback over shuffle', 'oil changed June 2026', 'usual commute is I-575 north'). Call this when the user says to remember something, states a lasting preference, or shares info worth recalling later. Do NOT use for one-off requests.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fact": {
                            "type": "string",
                            "description": "The fact to remember, phrased concisely as a standalone statement"
                        }
                    },
                    "required": ["fact"]
                }
            }
        ]

    # ------------------------------------------------------------------
    # System prompt (static — cached across interactions; the current time
    # is injected into the user turn instead, so the cache never breaks)
    # ------------------------------------------------------------------

    def _build_system_blocks(self) -> List[Dict]:
        vehicle = self.config.vehicle

        text = (
            "You are VIVIAN — a 1969 Ford Mustang Convertible. You ARE the car, not a program inside it. "
            f"Your engine is a {vehicle['engine']}, your transmission is a {vehicle['transmission']}, "
            f"and your rear end is a {vehicle['rear_end']}. "
            "Always speak in first person as the car itself. Be direct and concise. No sarcasm, no sign-offs, no filler. "

            "RULES — follow these exactly: "

            "1. TOOL CALLS FIRST. If a tool exists for the request, call it. "
            "NEVER answer from your training data when a tool can provide the answer. "
            "NEVER describe what you would do — just call the tool. "
            "NEVER say 'activating sentry mode' or 'let me search' as text — call the tool instead. "

            "2. WEB_SEARCH is MANDATORY for: news, current events, sports scores, stock prices, "
            "release dates, anything that changes over time. Do NOT answer these from memory. "

            "3. WHISPER TRANSCRIPTION ERRORS. The user's speech is transcribed by Whisper, which "
            "frequently garbles words. Interpret phonetic intent, not literal spelling. "
            "'sentry mode' (call activate_sentry_mode) may arrive as: century mode, centring mode, "
            "sentry med, sentinel mode, centro mode, sentromide, sentro mode, some trim out, "
            "some treatment, century mod. Any phonetic resemblance to a known command executes it. "

            "4. RESPONSE FORMAT: Keep your responses 3-4 sentences maximum unless it is implied "
            "that a longer response is needed. Spell out numbers and symbols for speech "
            "(seventy two degrees, three PM). No emoji. "

            "5. SUMMARIZE TOOL RESULTS — never read them back verbatim. Give what was asked "
            "for and drop secondary details unless the user asks for them. "

            "6. When the user tells you to remember something, or states a "
            "lasting preference or fact worth recalling on future drives, call "
            "remember_fact. Facts you've saved appear below. "

            "The current time is provided at the start of each user message. "
            "Base location: Woodstock, Georgia (use GPS for actual location). "
        )

        # Block 1 (persona + rules) is stable and carries the cache breakpoint.
        blocks = [{
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }]

        # Block 2 (remembered facts) is volatile — placed AFTER the cache
        # breakpoint so saving a fact doesn't invalidate the cached persona.
        facts = self._load_facts()
        if facts:
            blocks.append({
                "type": "text",
                "text": "Facts you have remembered:\n" + "\n".join(f"- {f}" for f in facts),
            })
        return blocks

    def _load_facts(self) -> List[str]:
        try:
            if self.facts_file.exists():
                return [ln.strip() for ln in self.facts_file.read_text().splitlines() if ln.strip()]
        except Exception as e:
            logger.warning(f"Could not read facts file: {e}")
        return []

    def _save_fact(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        facts = self._load_facts()
        # De-dupe (case-insensitive) and cap the list, dropping oldest
        if fact.lower() in (f.lower() for f in facts):
            return
        facts.append(fact)
        facts = facts[-self.max_facts:]
        try:
            tmp = str(self.facts_file) + ".tmp"
            with open(tmp, "w") as f:
                f.write("\n".join(facts) + "\n")
            os.replace(tmp, self.facts_file)
            logger.info(f"Remembered fact: {fact}")
        except Exception as e:
            logger.error(f"Could not save fact: {e}")

    def _time_prefix(self) -> str:
        now = datetime.now(self.timezone)
        return f"(Current time: {now.strftime('%Y-%m-%d %I:%M %p %Z')})"

    # ------------------------------------------------------------------
    # WAV playback helper (pre-cached announcement files)
    # ------------------------------------------------------------------

    def _play_wav(self, wav_path: str, device: str = None, volume: float = None) -> None:
        """Play a pre-recorded WAV file via aplay. Defaults to TTS output device.

        Args:
            volume: Optional 0.0-1.0 multiplier. When set, pipes through ffmpeg
                    to attenuate the audio before sending to aplay.
        """
        path = Path(wav_path)
        if not path.exists():
            logger.warning(f"WAV file not found: {wav_path}")
            return
        try:
            if device is None:
                device = getattr(self.tts, 'output_device', 'stereo_direct')
            if volume is not None:
                # Pipe through ffmpeg to adjust volume, then into aplay.
                # stderr goes to DEVNULL rather than an undrained PIPE: nobody
                # reads it, so a chatty ffmpeg could block writing to it.
                ffmpeg = subprocess.Popen(
                    ["ffmpeg", "-v", "quiet", "-i", str(path), "-af", f"volume={volume}",
                     "-f", "wav", "-"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                aplay = subprocess.Popen(
                    ["aplay", "-D", device],
                    stdin=ffmpeg.stdout, stderr=subprocess.PIPE
                )
                # The parent MUST drop its own copy of the read end. Otherwise,
                # if aplay exits early (device busy — the sentry speaker is
                # shared with TTS — or a bad device name), ffmpeg never gets
                # SIGPIPE, blocks forever writing into a full pipe, and the
                # old unbounded ffmpeg.wait() deadlocked the MAIN thread inside
                # the wake callback: sentry never armed and VIVIAN went deaf.
                ffmpeg.stdout.close()
                try:
                    aplay.communicate(timeout=30)
                finally:
                    for proc in (aplay, ffmpeg):
                        if proc.poll() is None:
                            proc.kill()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning("WAV playback: child did not exit after kill")
            else:
                subprocess.run(
                    ["aplay", "-D", device, str(path)],
                    timeout=30,
                    stderr=subprocess.PIPE
                )
        except Exception as e:
            logger.error(f"WAV playback error: {e}")

    # ------------------------------------------------------------------
    # Diagnostics (unchanged from Mk17 — pure shell, no API involvement)
    # ------------------------------------------------------------------

    def _run_diagnostics(self, diag_type: str) -> str:
        """Run system/network diagnostics on the Pi and return results as text."""

        def cmd(command, timeout=6):
            try:
                out = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=timeout
                )
                return (out.stdout.strip() or out.stderr.strip()) or "no output"
            except subprocess.TimeoutExpired:
                return "timed out"
            except Exception as e:
                return f"error: {e}"

        lines = []

        if diag_type in ("network", "full"):
            lines.append("=== NETWORK ===")

            # Internet connectivity
            ping = cmd("ping -c 2 -W 2 8.8.8.8 2>&1 | tail -2")
            lines.append(f"Internet (ping 8.8.8.8):\n{ping}")

            # Interface IPs
            eth0_ip  = cmd("ip addr show eth0  | awk '/inet /{print $2}'")
            wlan0_ip = cmd("ip addr show wlan0 | awk '/inet /{print $2}'")
            lines.append(f"eth0  (cellular): {eth0_ip  or 'no IP'}")
            lines.append(f"wlan0 (VIVIAN AP): {wlan0_ip or 'no IP'}")

            # Default route
            route = cmd("ip route show default")
            lines.append(f"Default route: {route}")

            # AP clients
            client_count = cmd("iw dev wlan0 station dump 2>/dev/null | grep -c '^Station' || echo 0")
            lines.append(f"AP clients connected: {client_count}")

            leases = cmd(
                "awk '{print $4\" (\"$3\")\"}' "
                "/var/lib/NetworkManager/dnsmasq-wlan0.leases 2>/dev/null | head -5"
            )
            if leases and leases != "no output":
                lines.append(f"Client hostnames/IPs: {leases}")

            # Cellular modem connectivity
            modem_gw = cmd("ip route show default dev eth0 | awk '/via/{print $3}'")
            if modem_gw and modem_gw != "no output":
                modem_ping = cmd(f"ping -c 1 -W 2 {modem_gw} 2>&1 | tail -1")
                lines.append(f"Cellular modem gateway ({modem_gw}): {modem_ping}")

        if diag_type in ("system", "full"):
            lines.append("=== SYSTEM ===")

            # CPU temperature
            temp = cmd(
                "vcgencmd measure_temp 2>/dev/null || "
                "awk '{printf \"%.1f°C\", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null"
            )
            lines.append(f"CPU temp: {temp}")

            # Uptime
            uptime = cmd("uptime -p")
            lines.append(f"Uptime: {uptime}")

            # Load average
            load = cmd("awk '{print $1\" (1m)  \"$2\" (5m)  \"$3\" (15m)\"}' /proc/loadavg")
            lines.append(f"CPU load: {load}")

            # RAM
            ram = cmd("free -m | awk '/^Mem:/{print $3\"MB used / \"$2\"MB total\"}'")
            lines.append(f"RAM: {ram}")

            # Disk
            disk = cmd("df -h / | awk 'NR==2{print $3\" used / \"$2\" total (\"$5\" full)\"}'")
            lines.append(f"Disk (/): {disk}")

            # VIVIAN process
            vivian = cmd("pgrep -a -f 'python.*main.py' | grep -v grep | head -2")
            lines.append(f"VIVIAN process: {vivian or 'not found'}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Streaming API call — speaks text sentences as they arrive
    # ------------------------------------------------------------------

    @staticmethod
    def _flush_sentences(buffer: str):
        """Pull complete sentences off a growing text buffer.

        Returns (sentences, remaining). Only flushes on end-punctuation
        followed by whitespace, so decimals ('72.5') stay intact.
        """
        sentences = []
        while True:
            m = re.match(r"\s*(.+?[.!?])(\s+)", buffer, re.S)
            if not m:
                break
            sentences.append(m.group(1).strip())
            buffer = buffer[m.end():]
        return sentences, buffer

    def _stream_and_speak(self, messages: List[Dict]):
        """Stream one model turn. Speak text sentences as they arrive (only
        on a pure text turn — tool turns emit no narration per the system
        prompt). Returns (final_message, spoke_any)."""
        speaker = None
        buffer = ""
        saw_tool_use = False
        spoke_any = False
        # The outer try/finally is what guarantees the stream speaker is torn
        # down. Closing it only in the anthropic-specific handlers left every
        # OTHER exception orphaning the synth/play threads and their live piper
        # and aplay children — e.g. get_final_message() raises AssertionError
        # (not an APIError) when the SSE stream ends without message_stop. The
        # orphan kept playing the partial answer on the same ALSA device that
        # main.py then used for "I encountered an error", garbling both.
        try:
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=500,
                    system=self._build_system_blocks(),
                    tools=self.tools,
                    messages=messages,
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_start" and \
                                getattr(event.content_block, "type", None) == "tool_use":
                            saw_tool_use = True
                        elif event.type == "content_block_delta" and \
                                getattr(event.delta, "type", None) == "text_delta":
                            if saw_tool_use:
                                continue  # don't narrate a tool turn
                            buffer += event.delta.text
                            sentences, buffer = self._flush_sentences(buffer)
                            for s in sentences:
                                if speaker is None:
                                    speaker = self.tts.stream_speaker()
                                speaker.feed(s)
                                spoke_any = True
                    final = stream.get_final_message()
            except anthropic.RateLimitError as e:
                self._last_error = f"429 rate limit: {e}"
                logger.error(self._last_error)
                raise RuntimeError(self._last_error)
            except anthropic.APIConnectionError as e:
                self._last_error = f"Connection error: {e}"
                logger.error(self._last_error)
                raise RuntimeError(self._last_error)
            except anthropic.APIStatusError as e:
                self._last_error = f"API error {e.status_code}: {e.message}"
                logger.error(self._last_error)
                raise RuntimeError(self._last_error)

            # Flush the trailing partial sentence on a text turn
            if not saw_tool_use and buffer.strip():
                if speaker is None:
                    speaker = self.tts.stream_speaker()
                speaker.feed(buffer.strip())
                spoke_any = True
        finally:
            if speaker is not None:
                speaker.close()  # blocks until playback finishes

        return final, spoke_any

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _handle_fast_path(self, name: str, args: Dict, user_prompt: str) -> None:
        """Execute a fast-path tool: act, optionally speak, end the turn."""
        logger.info(f"Fast-path tool: {name} with args: {args}")

        if name == "control_music":
            action_map = {
                "skip": self.spotify.skip,
                "previous": self.spotify.previous,
                "pause": self.spotify.pause,
                "resume": self.spotify.resume,
                "play_playlist": self.spotify.play_playlist,
            }
            action = args.get("action")
            ok = True
            if action in action_map:
                ok = action_map[action]()
            if action == "play_playlist":
                announcement = ("Starting your driving playlist." if ok
                                else "I couldn't start your playlist — Spotify may be offline.")
                self.tts.speak(announcement)
            self.memory.add_interaction(user_prompt, f"[Tool: control_music/{action}]")
            return

        if name == "play_track":
            track_name = args["track_name"]
            artist_name = args.get("artist_name", "")
            track_info = self.spotify.play_track(
                track_name, artist_name, start_immediately=False
            )
            if not track_info:
                # Speech-to-text splits numbers in titles ("in the year 25-25"
                # for "In the Year 2525"), which Spotify's search will not
                # resolve. Retry once with the digits rejoined before giving up.
                retry = re.sub(r"(?<=\d)[\s-]+(?=\d)", "", track_name)
                if retry != track_name:
                    logger.info(f"Track not found, retrying with joined digits: "
                                f"'{track_name}' -> '{retry}'")
                    track_info = self.spotify.play_track(
                        retry, artist_name, start_immediately=False
                    )
            if not track_info:
                # Say so rather than failing silently — the driver otherwise has
                # no idea whether she heard them at all.
                self.tts.speak(f"I couldn't find {track_name} on Spotify.")
                return
            if track_info:
                announcement = f"Playing {track_info['name']} by {track_info['artists']}."
                self.memory.add_interaction(user_prompt, announcement)
                self.tts.speak(announcement)
                time.sleep(0.3)
                self.spotify.resume()
            return

        if name == "play_album":
            result = self.spotify.play_album(
                args["album_name"], args["artist_name"], start_immediately=False
            )
            if result:
                announcement = f"Playing the album {result}."
                self.memory.add_interaction(user_prompt, announcement)
                self.tts.speak(announcement)
                time.sleep(0.3)
                self.spotify.resume()
            return

        if name == "play_artist":
            artist_info = self.spotify.play_artist_top_tracks(
                args["artist_name"], start_immediately=False
            )
            if artist_info:
                announcement = f"Playing top songs by {artist_info['name']}."
                self.memory.add_interaction(user_prompt, announcement)
                self.tts.speak(announcement)
                time.sleep(0.3)
                self.spotify.resume()
            return

        if name == "reboot_system":
            self.memory.add_interaction(user_prompt, "[System: Reboot]")
            self.tts.speak("Authorization granted. Initiating system reboot, please stand by.")
            time.sleep(1)
            os.system("sudo /sbin/reboot")
            return

        if name == "shutdown_system":
            self.memory.add_interaction(user_prompt, "[System: Shutdown]")
            self.tts.speak("Shutting down now. Safe to cut power in a few seconds.")
            time.sleep(1)
            os.system("sudo /sbin/shutdown now")
            return

        if name == "activate_sentry_mode":
            self.sentry_requested = True
            wav = Path(__file__).parent / 'Sentry' / 'Sentry_Activation.wav'
            sentry_device = self.config.sentry.get('tts_output', 'internal_plug')
            self._play_wav(str(wav), device=sentry_device, volume=0.35)
            self.memory.add_interaction(user_prompt, "[Sentry: Activated]")
            return

        if name == "remember_fact":
            fact = args.get("fact", "")
            self._save_fact(fact)
            announcement = "Got it. I'll remember that."
            self.memory.add_interaction(user_prompt, announcement)
            self.tts.speak(announcement)
            time.sleep(0.3)
            return

    def _execute_data_tool(self, name: str, args: Dict) -> str:
        """Execute a data tool and return its result text for the model."""
        logger.info(f"Executing tool: {name} with args: {args}")

        if name == "get_current_song":
            playback_info = self.spotify.get_current_playback()
            if playback_info:
                return self.spotify.format_playback_info(playback_info)
            return "No music is currently playing."

        if name == "get_weather":
            return self.system_info.get_detailed_weather(
                args.get("location"), args.get("query_type", "current")
            )

        if name == "get_current_location":
            if self.system_info.gps and self.system_info.gps.has_fix():
                detailed_addr = self.system_info.gps.get_detailed_address()
                if detailed_addr:
                    address_parts = []
                    if detailed_addr['house_number'] and detailed_addr['road']:
                        address_parts.append(f"{detailed_addr['house_number']} {detailed_addr['road']}")
                    elif detailed_addr['road']:
                        address_parts.append(detailed_addr['road'])
                    if detailed_addr['city']:
                        address_parts.append(detailed_addr['city'])
                    if detailed_addr['state']:
                        address_parts.append(detailed_addr['state'])
                    if detailed_addr['postcode']:
                        address_parts.append(detailed_addr['postcode'])

                    location_str = "Current location approximately: " + ", ".join(address_parts)
                    location_str += (". Note: GPS coordinates are accurate to within 5-10 meters; "
                                     "street address may vary slightly.")
                    return location_str
                return ("GPS has a fix but could not determine street address. You are near " +
                        str(self.system_info.gps.get_city_state()))
            return "GPS is not available or does not have a fix yet."

        if name == "find_nearby_places":
            place_category = args["place_category"]
            limit = args.get("limit", 5)

            if not self.system_info.gps or not self.system_info.gps.has_fix():
                return "GPS is not available or does not have a fix yet. Cannot search for nearby places."

            category_map = {
                "gas_station": ("amenity", "fuel"),
                "restaurant": ("amenity", "restaurant"),
                "fast_food": ("amenity", "fast_food"),
                "cafe": ("amenity", "cafe"),
                "hotel": ("tourism", "hotel"),
                "atm": ("amenity", "atm"),
                "bank": ("amenity", "bank"),
                "pharmacy": ("amenity", "pharmacy"),
                "hospital": ("amenity", "hospital"),
                "police": ("amenity", "police"),
                "parking": ("amenity", "parking"),
                "charging_station": ("amenity", "charging_station")
            }

            if place_category not in category_map:
                return f"Unknown place category: {place_category}"

            place_type, search_query = category_map[place_category]
            places = self.system_info.gps.find_nearby_places(
                place_type=place_type, search_query=search_query, limit=limit
            )

            if places:
                results_text = f"Found {len(places)} nearby {place_category.replace('_', ' ')}s:\n"
                for i, place in enumerate(places, 1):
                    results_text += f"{i}. {place['name']} - {place['distance_miles']} miles away"
                    if place['address'] != 'Address not available':
                        results_text += f" at {place['address']}"
                    results_text += "\n"
                return results_text
            return f"No {place_category.replace('_', ' ')}s found within 10 miles."

        if name == "find_specific_business":
            business_name = args["business_name"]

            if not self.system_info.gps or not self.system_info.gps.has_fix():
                return "GPS is not available or does not have a fix yet. Cannot search for businesses."

            places = self.system_info.gps.find_specific_business(business_name)
            if places:
                results_text = f"Found {len(places)} {business_name} location(s) nearby:\n"
                for i, place in enumerate(places, 1):
                    results_text += f"{i}. {place['name']} - {place['distance_miles']} miles away"
                    if place['address'] != 'Address not available':
                        results_text += f" at {place['address']}"
                    results_text += "\n"
                return results_text
            return f"No {business_name} locations found within 10 miles."

        if name == "run_diagnostics":
            return self._run_diagnostics(args.get("type", "full"))

        logger.warning(f"Unknown tool requested: {name}")
        return f"Unknown tool: {name}"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_request(self, user_prompt: str) -> None:
        """Process user request and generate a spoken response"""
        logger.info(f"User: {user_prompt}")

        # Session context: last 3 exchanges, then the current message with
        # the (volatile) timestamp inline so the cached system prompt stays
        # byte-identical across interactions.
        messages: List[Dict] = []
        for ctx in self.memory.get_session_context()[-3:]:
            messages.append({"role": "user", "content": ctx["user"]})
            messages.append({"role": "assistant", "content": ctx["assistant"]})
        messages.append({
            "role": "user",
            "content": f"{self._time_prefix()}\n{user_prompt}",
        })

        last_tool_results: List[str] = []

        for _ in range(MAX_TOOL_ITERATIONS):
            # Stream the turn — text sentences are spoken as they arrive.
            response, spoke_any = self._stream_and_speak(messages)

            if response.stop_reason == "refusal":
                logger.warning("Model refused the request")
                text = "I can't help with that one."
                self.memory.add_interaction(user_prompt, text)
                self.tts.speak(text)
                return

            # Server-side tool (web search) paused mid-loop — resume
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if response.stop_reason == "tool_use" and tool_uses:
                # Fast-path tools end the turn without another API round-trip —
                # but ONLY when the fast-path tool is the whole turn. When the
                # model batches several tools in one turn ("play some Led
                # Zeppelin and tell me the weather"), returning here silently
                # discarded every other tool call: the music started and the
                # weather question was never answered, with nothing logged.
                fast = next((b for b in tool_uses if b.name in FAST_PATH_TOOLS), None)
                if fast is not None and len(tool_uses) == 1:
                    self._handle_fast_path(fast.name, dict(fast.input), user_prompt)
                    return

                # Data tools: execute all, return results in ONE user message
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in tool_uses:
                    try:
                        if block.name in FAST_PATH_TOOLS:
                            # Mixed turn: perform the action (and its spoken
                            # confirmation) inline, then report it as an
                            # ordinary tool result so the model can go on to
                            # answer the rest of the request.
                            self._handle_fast_path(block.name, dict(block.input), user_prompt)
                            content = f"{block.name} completed."
                        else:
                            content = self._execute_data_tool(block.name, dict(block.input))
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                        })
                        last_tool_results.append(content)
                    except Exception as e:
                        logger.error(f"Tool {block.name} failed: {e}")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool error: {e}",
                            "is_error": True,
                        })
                messages.append({"role": "user", "content": results})
                continue

            # Final text response — already spoken during streaming if spoke_any
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if not text and last_tool_results:
                logger.warning("Empty final response, falling back to raw tool result")
                text = last_tool_results[-1][:400]
            if not text:
                text = "I didn't catch that. Please try again."

            logger.info(f"VIVIAN: {text}")
            metadata = {
                "music_playing": self.spotify.load_state(),
                "model": self.model,
            }
            self.memory.add_interaction(user_prompt, text, metadata)
            if not spoke_any:
                # Streaming produced no audio (empty response / fallback text)
                self.tts.speak(text)
            time.sleep(0.3)  # brief tail before main.py unmutes/resumes
            return

        # Tool loop ran away — surface as an error via main.py's handler
        self._last_error = "Tool loop exceeded maximum iterations"
        raise RuntimeError(self._last_error)
