import spotipy
import requests
import threading
import time
import json
import os
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

SPOTIFY_SHARED_FILE = "/tmp/vivian_spotify.json"


class SpotifyController:
    """Manage Spotify playback and state"""
    
    def __init__(self, config):
        self.config = config
        self.client_id = config.spotify['client_id']
        self.client_secret = config.spotify['client_secret']
        self.refresh_token = config.spotify['refresh_token']
        self.device_name = config.spotify['device_name']
        self.default_playlist_uri = config.spotify.get(
            'default_playlist_uri', "spotify:playlist:3b3cb2OH1MgoFb2bzvRT9e"
        )
        self.access_token = None
        self.state_file = config.files['state_file']
        # Per-thread cached spotipy client (see get_client). _token_gen is
        # bumped on every refresh so cached clients pick up the new token.
        self._client_local = threading.local()
        self._token_gen = 0
        # Did the LAST refresh attempt work? Keying the retry interval off
        # "access_token is None" meant a failed refresh still waited the full
        # 30 minutes, i.e. up to half an hour of dead music control.
        self._last_refresh_ok = False

        # Initial token refresh — must NOT kill VIVIAN when the car starts
        # in a dead zone; the refresh thread retries until it succeeds.
        try:
            self.refresh_access_token()
        except Exception as e:
            logger.warning(
                f"Initial Spotify token refresh failed (offline at boot?): {e} "
                f"— continuing without Spotify, retrying in background"
            )

        # Start background token refresh
        self._start_token_refresh_thread()

        # Start background playback status writer for CRT display
        self._start_playback_writer_thread()

        logger.info("Spotify controller initialized")
    
    def refresh_access_token(self) -> str:
        """Refresh Spotify access token"""
        token_url = "https://accounts.spotify.com/api/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        
        try:
            response = requests.post(token_url, data=payload, timeout=10)
            
            # Log detailed error info for debugging
            if response.status_code != 200:
                logger.error(f"Spotify token refresh failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                logger.error(f"Check your Spotify credentials in config.yaml")
            
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            # Invalidate cached clients so the next call uses the new token
            self._token_gen += 1
            self._last_refresh_ok = True
            logger.info("Spotify token refreshed")
            return self.access_token
        except requests.RequestException as e:
            self._last_refresh_ok = False
            logger.error(f"Token refresh failed: {e}")
            logger.error("Common issues:")
            logger.error("1. Invalid refresh_token (may have expired)")
            logger.error("2. Incorrect client_id or client_secret")
            logger.error("3. Refresh token needs quotes in config.yaml")
            raise
    
    def _write_playback_shared_file(self):
        """Write current playback info to shared file for CRT display."""
        try:
            info = self.get_current_playback()
            if info:
                data = {
                    'is_playing': info['is_playing'],
                    'track': info['track_name'],
                    'artist': info['artists'],
                    'progress_ms': info['progress_ms'],
                    'duration_ms': info['duration_ms'],
                    'timestamp': time.time(),
                }
            else:
                data = {'is_playing': False, 'timestamp': time.time()}
            tmp = SPOTIFY_SHARED_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f)
            os.replace(tmp, SPOTIFY_SHARED_FILE)
        except Exception as e:
            logger.debug(f"Playback shared file write failed: {e}")

    def _start_playback_writer_thread(self, interval_seconds: int = 1):
        """Background thread that writes playback state to shared file."""
        def writer():
            while True:
                time.sleep(interval_seconds)
                try:
                    self._write_playback_shared_file()
                except Exception as e:
                    logger.debug(f"Playback writer error: {e}")

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        logger.info("Playback writer thread started")

    def _start_token_refresh_thread(self, interval_seconds: int = 1800):
        """Start background thread for periodic token refresh (every 30 minutes).

        Whenever the LAST attempt failed (offline boot, dead zone, expired
        token), retry every 60s so Spotify comes back as soon as the cellular
        link does. Keying this off access_token was wrong: the token is never
        None again after the first success, so a failed refresh used to wait
        the full 30 minutes.
        """
        def refresher():
            while True:
                time.sleep(interval_seconds if self._last_refresh_ok else 60)
                try:
                    self.refresh_access_token()
                except Exception as e:
                    logger.error(f"Background token refresh failed: {e}")

        thread = threading.Thread(target=refresher, daemon=True)
        thread.start()
        logger.info("Token refresh thread started")
    
    def get_client(self) -> spotipy.Spotify:
        """Get authenticated Spotify client.

        requests_timeout/retries are passed EXPLICITLY: spotipy is unpinned and
        its defaults have changed between versions (older ones had NO timeout,
        i.e. an unbounded call on the wake-word path with the driver waiting).
        Short timeout, no retries — failing fast beats stalling the interaction.

        The client is cached per thread and reused instead of rebuilt on every
        call (the playback writer alone called this once a second, each time
        creating a new TLS session and connection pool). Cached per THREAD
        rather than globally so the writer, display poller and command path
        never share one requests.Session. A token refresh bumps _token_gen,
        which retires every cached client.
        """
        local = self._client_local
        if getattr(local, 'gen', None) != self._token_gen:
            local.client = spotipy.Spotify(
                auth=self.access_token,
                requests_timeout=3,
                retries=0,
            )
            local.gen = self._token_gen
        return local.client

    def _refreshed_on_401(self, e: Exception) -> bool:
        """Refresh the token once if a Spotify call failed with HTTP 401.

        Returns True when the token was refreshed and the call is worth
        retrying. Without this an expired token stays dead until the next
        periodic refresh tick instead of healing on first use.
        """
        if not isinstance(e, spotipy.SpotifyException) or getattr(e, 'http_status', None) != 401:
            return False
        logger.warning("Spotify returned 401 (expired token) — refreshing and retrying once")
        try:
            self.refresh_access_token()
            return True
        except Exception as refresh_err:
            logger.error(f"Token refresh after 401 failed: {refresh_err}")
            return False

    def _devices(self) -> list:
        """List Spotify devices, self-healing once on an expired token."""
        try:
            return self.get_client().devices().get("devices", [])
        except Exception as e:
            if not self._refreshed_on_401(e):
                raise
            return self.get_client().devices().get("devices", [])

    def _current_playback(self) -> Optional[Dict]:
        """Fetch raw current playback, self-healing once on an expired token."""
        try:
            return self.get_client().current_playback()
        except Exception as e:
            if not self._refreshed_on_401(e):
                raise
            return self.get_client().current_playback()

    def get_device_id(self) -> Optional[str]:
        """Get active device or fallback to named device"""
        try:
            devices = self._devices()

            # Try active device first
            for device in devices:
                if device.get("is_active"):
                    logger.debug(f"Found active device: {device['name']}")
                    return device["id"]
            
            # Fallback to named device
            for device in devices:
                if device["name"].lower() == self.device_name.lower():
                    logger.debug(f"Found named device: {device['name']}")
                    return device["id"]
            
            logger.warning("No suitable Spotify device found")
            return None
        except Exception as e:
            logger.error(f"Error getting device ID: {e}")
            return None
    
    def save_state(self, should_resume: bool):
        """Save music resume state"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({"music_should_resume": should_resume}, f)
            logger.debug(f"State saved: should_resume={should_resume}")
        except IOError as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self) -> bool:
        """Load music resume state"""
        if not os.path.exists(self.state_file):
            return True
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f).get("music_should_resume", True)
                logger.debug(f"State loaded: should_resume={state}")
                return state
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading state: {e}")
            return True
    
    def pause(self) -> bool:
        """Pause playback (explicit user command - prevents auto-resume)"""
        device_id = self.get_device_id()
        if not device_id:
            logger.warning("No device found for pause")
            return False

        try:
            sp = self.get_client()
            sp.pause_playback(device_id=device_id)
            self.save_state(False)
            logger.info("Playback paused (user requested)")
            return True
        except Exception as e:
            logger.error(f"Pause failed: {e}")
            return False

    def resume(self) -> bool:
        """Resume playback (explicit user command - enables auto-resume)"""
        device_id = self.get_device_id()
        if not device_id:
            logger.warning("No device found for resume")
            return False

        try:
            sp = self.get_client()
            sp.start_playback(device_id=device_id)
            self.save_state(True)
            logger.info("Playback resumed (user requested)")
            return True
        except Exception as e:
            logger.error(f"Resume failed: {e}")
            return False

    def soft_pause(self) -> bool:
        """Pause playback without changing state (temporary, for mute during interaction).

        Does not modify the state file so auto-resume still knows whether
        music should come back after the interaction.
        """
        try:
            device_id = self.get_device_id()
            if not device_id:
                return False
            sp = self.get_client()
            sp.pause_playback(device_id=device_id)
            logger.info("Playback soft-paused (interaction)")
            return True
        except Exception as e:
            logger.debug(f"Soft pause failed (may not be playing): {e}")
            return False

    def soft_resume(self) -> bool:
        """Resume playback only if state says music should be playing.

        Checks the state file first — if the user explicitly paused,
        this will NOT resume. Does not modify the state file.
        """
        if not self.load_state():
            logger.info("Soft resume skipped (music was explicitly paused)")
            return False
        try:
            device_id = self.get_device_id()
            if not device_id:
                logger.warning("Soft resume: no device found")
                return False
            sp = self.get_client()
            sp.start_playback(device_id=device_id)
            logger.info("Playback soft-resumed (interaction over)")
            return True
        except Exception as e:
            logger.debug(f"Soft resume failed (may already be playing): {e}")
            return False
    
    def skip(self) -> bool:
        """Skip to next track"""
        device_id = self.get_device_id()
        if not device_id:
            logger.warning("No device found for skip")
            return False
        
        try:
            sp = self.get_client()
            sp.next_track(device_id=device_id)
            sp.start_playback(device_id=device_id)
            self.save_state(True)
            time.sleep(0.5)
            self._write_playback_shared_file()
            logger.info("Skipped to next track")
            return True
        except Exception as e:
            logger.error(f"Skip failed: {e}")
            return False
    
    def previous(self) -> bool:
        """Go to previous track"""
        device_id = self.get_device_id()
        if not device_id:
            logger.warning("No device found for previous")
            return False
        
        try:
            sp = self.get_client()
            sp.previous_track(device_id=device_id)
            sp.start_playback(device_id=device_id)
            self.save_state(True)
            time.sleep(0.5)
            self._write_playback_shared_file()
            logger.info("Went to previous track")
            return True
        except Exception as e:
            logger.error(f"Previous failed: {e}")
            return False
    
    def play_playlist(self, playlist_uri: str = None) -> bool:
        """Play a specific playlist from the beginning (default from config)"""
        if playlist_uri is None:
            playlist_uri = self.default_playlist_uri
        device_id = self.get_device_id()
        if not device_id:
            logger.warning("No device found for playlist")
            return False

        try:
            sp = self.get_client()
            sp.start_playback(device_id=device_id, context_uri=playlist_uri)
            self.save_state(True)
            logger.info(f"Playing playlist: {playlist_uri}")
            return True
        except Exception as e:
            logger.error(f"Playlist playback failed: {e}")
            return False

    def toggle_shuffle(self) -> Optional[bool]:
        """Toggle shuffle state. Returns new shuffle state or None on failure."""
        device_id = self.get_device_id()
        if not device_id:
            return None
        try:
            sp = self.get_client()
            playback = sp.current_playback()
            if not playback:
                return None
            current = playback.get('shuffle_state', False)
            new_state = not current
            sp.shuffle(state=new_state, device_id=device_id)
            logger.info(f"Shuffle {'on' if new_state else 'off'}")
            return new_state
        except Exception as e:
            logger.error(f"Toggle shuffle failed: {e}")
            return None
    
    def play_track(self, track_name: str, artist_name: str = "", start_immediately: bool = False) -> Optional[Dict[str, str]]:
        """
        Search and play a specific track
        
        Args:
            track_name: Name of the track
            artist_name: Name of the artist (optional)
            start_immediately: If False, queues track but doesn't start playback (default: False)
        
        Returns:
            dict with track info if successful, None otherwise
        """
        sp = self.get_client()
        
        # Simple search queries — mirrors what the Spotify app does
        queries = []
        if track_name and artist_name:
            queries.append(f'{track_name} {artist_name}')
        queries.append(track_name)
        
        # Keywords that indicate a non-studio version
        _skip_keywords = {'live', 'concert', 'karaoke', 'tribute', 'cover',
                          'remix', 'demo', 'rough mix', 'session', 'unplugged',
                          'instrumental', 'bonus track', 'made famous'}

        def _clean_name(name):
            """Strip parenthetical suffixes for comparison: 'Seek & Destroy (Remastered)' -> 'seek & destroy'"""
            import re
            return re.sub(r'\s*[\(\[\-].*$', '', name).strip().lower()

        def _names_match(result_name, query_name):
            """Check if track name matches what was requested (ignoring suffixes and & vs and)."""
            a = _clean_name(result_name).replace('&', 'and')
            b = _clean_name(query_name).replace('&', 'and')
            return a == b or a in b or b in a

        def _is_studio(track):
            """Check if track appears to be the original studio version."""
            name = track.get("name", "").lower()
            album = track.get("album", {}).get("name", "").lower()
            album_type = track.get("album", {}).get("album_type", "")
            artists = " ".join(a.get("name", "").lower() for a in track.get("artists", []))
            for kw in _skip_keywords:
                if kw in name or kw in album or kw in artists:
                    return False
            if album_type == "compilation":
                return False
            if artist_name and artist_name.lower() not in artists:
                return False
            return True

        # Try each query, preferring studio versions that match the track name
        chosen_track = None
        fallback_track = None
        for query in queries:
            try:
                results = sp.search(q=query, type="track", limit=20)
                items = results.get("tracks", {}).get("items", [])
                if not items:
                    continue
                if fallback_track is None:
                    fallback_track = items[0]
                # Filter to studio versions that actually match the requested track name
                studio = [t for t in items
                          if _is_studio(t) and _names_match(t.get("name", ""), track_name)]
                if studio:
                    studio.sort(key=lambda t: t.get("popularity", 0), reverse=True)
                    chosen_track = studio[0]
                    logger.info(f"Selected studio version: {chosen_track['name']} "
                                f"(popularity: {chosen_track.get('popularity', 0)}, "
                                f"album: {chosen_track['album']['name']})")
                    break
            except Exception as e:
                logger.error(f"Search error for '{query}': {e}")

        if not chosen_track:
            chosen_track = fallback_track
        
        if not chosen_track:
            logger.warning(f"Track not found: {track_name} by {artist_name}")
            return None
        
        # Get device
        device_id = self.get_device_id()
        if not device_id:
            logger.warning("No device found for track playback")
            return None
        
        try:
            track_uri = chosen_track.get("uri")
            
            if start_immediately:
                # Start playing immediately
                sp.start_playback(device_id=device_id, uris=[track_uri])
            else:
                # Queue track but pause immediately so announcement can finish
                sp.start_playback(device_id=device_id, uris=[track_uri])
                time.sleep(0.5)  # Small delay to ensure playback started
                sp.pause_playback(device_id=device_id)
            
            self.save_state(True)
            self._write_playback_shared_file()

            track_info = {
                "name": chosen_track.get("name", track_name),
                "artists": ", ".join(a.get("name", "") for a in chosen_track.get("artists", []))
            }
            logger.info(f"Queued: {track_info['name']} by {track_info['artists']}")
            return track_info
        except Exception as e:
            logger.error(f"Track playback failed: {e}")
            return None
    
    def play_album(self, album_name: str, artist_name: str, start_immediately: bool = False) -> Optional[str]:
        """
        Search and play a specific album
        
        Args:
            album_name: Name of the album
            artist_name: Name of the artist
            start_immediately: If False, queues album but doesn't start playback (default: False)
        
        Returns:
            album description if successful, None otherwise
        """
        sp = self.get_client()
        
        try:
            results = sp.search(
                q=f"album:{album_name} artist:{artist_name}",
                type="album",
                limit=1
            )
            albums = results.get("albums", {}).get("items", [])
            
            if not albums:
                logger.warning(f"Album not found: {album_name} by {artist_name}")
                return None
            
            album_uri = albums[0]["uri"]
            device_id = self.get_device_id()
            
            if not device_id:
                logger.warning("No device found for album playback")
                return None
            
            # Disable shuffle for album playback
            sp.shuffle(state=False, device_id=device_id)
            
            if start_immediately:
                sp.start_playback(device_id=device_id, context_uri=album_uri)
            else:
                # Queue album but pause immediately
                sp.start_playback(device_id=device_id, context_uri=album_uri)
                time.sleep(0.5)
                sp.pause_playback(device_id=device_id)
            
            self.save_state(True)
            
            album_desc = f"{album_name} by {artist_name}"
            logger.info(f"Queued album: {album_desc}")
            return album_desc
        except Exception as e:
            logger.error(f"Album playback failed: {e}")
            return None
    
    def play_artist_top_tracks(self, artist_name: str, start_immediately: bool = False) -> Optional[Dict[str, str]]:
        """
        Play an artist's top tracks
        
        Args:
            artist_name: Name of the artist
            start_immediately: If False, queues tracks but doesn't start playback (default: False)
        
        Returns:
            dict with artist info if successful, None otherwise
        """
        sp = self.get_client()
        
        try:
            # Search for the artist
            results = sp.search(q=f'artist:"{artist_name}"', type="artist", limit=1)
            artists = results.get("artists", {}).get("items", [])
            
            if not artists:
                logger.warning(f"Artist not found: {artist_name}")
                return None
            
            artist = artists[0]
            artist_id = artist["id"]
            artist_actual_name = artist["name"]
            
            # Get artist's top tracks (US market)
            top_tracks = sp.artist_top_tracks(artist_id, country="US")
            tracks = top_tracks.get("tracks", [])
            
            if not tracks:
                logger.warning(f"No top tracks found for {artist_actual_name}")
                return None
            
            # Get URIs of top tracks (up to 10)
            track_uris = [track["uri"] for track in tracks[:10]]
            
            device_id = self.get_device_id()
            if not device_id:
                logger.warning("No device found for artist playback")
                return None
            
            # Play the top tracks
            if start_immediately:
                sp.start_playback(device_id=device_id, uris=track_uris)
            else:
                # Queue tracks but pause immediately
                sp.start_playback(device_id=device_id, uris=track_uris)
                time.sleep(0.5)
                sp.pause_playback(device_id=device_id)
            
            self.save_state(True)
            
            artist_info = {
                "name": artist_actual_name,
                "track_count": len(track_uris)
            }
            logger.info(f"Queued {artist_info['track_count']} top tracks by {artist_actual_name}")
            return artist_info
        except Exception as e:
            logger.error(f"Artist playback failed: {e}")
            return None
    
    def get_current_playback(self) -> Optional[Dict]:
        """Get current playback information"""
        try:
            playback = self._current_playback()

            if not playback or not playback.get("item"):
                return None
            
            item = playback["item"]
            
            # Extract relevant information
            images = item.get("album", {}).get("images", [])
            art_url = images[0]["url"] if images else None

            current_info = {
                "is_playing": playback.get("is_playing", False),
                "track_name": item.get("name", "Unknown"),
                "artists": ", ".join(artist.get("name", "") for artist in item.get("artists", [])),
                "album": item.get("album", {}).get("name", "Unknown"),
                "album_uri": item.get("album", {}).get("uri", ""),
                "album_art_url": art_url,
                "track_uri": item.get("uri", ""),
                "duration_ms": item.get("duration_ms", 0),
                "progress_ms": playback.get("progress_ms", 0),
                "device_name": playback.get("device", {}).get("name", "Unknown"),
                "volume": playback.get("device", {}).get("volume_percent", 0),
                "shuffle": playback.get("shuffle_state", False),
                "repeat": playback.get("repeat_state", "off")
            }
            
            # Calculate percentage played
            if current_info["duration_ms"] > 0:
                current_info["percent_played"] = int(
                    (current_info["progress_ms"] / current_info["duration_ms"]) * 100
                )
            else:
                current_info["percent_played"] = 0
            
            # Calculate time remaining
            current_info["remaining_ms"] = current_info["duration_ms"] - current_info["progress_ms"]
            
            logger.debug(f"Current playback: {current_info['track_name']} by {current_info['artists']}")
            return current_info
            
        except Exception as e:
            logger.error(f"Error getting current playback: {e}")
            return None
    
    def get_album_art_url(self) -> Optional[str]:
        """Get the album art URL for the currently playing track."""
        try:
            sp = self.get_client()
            playback = sp.current_playback()
            if not playback or not playback.get("item"):
                return None
            images = playback["item"].get("album", {}).get("images", [])
            return images[0]["url"] if images else None
        except Exception as e:
            logger.error(f"Error getting album art URL: {e}")
            return None

    def play_album_at_position(self, album_uri: str, position: int = 0) -> bool:
        """Play an album starting at a specific track position.

        Used by the Spotify touch display to resume an album from a
        particular track so playback continues through the rest of the album.
        """
        device_id = self.get_device_id()
        if not device_id:
            return False
        try:
            sp = self.get_client()
            sp.shuffle(state=False, device_id=device_id)
            sp.start_playback(
                device_id=device_id,
                context_uri=album_uri,
                offset={"position": position},
            )
            self.save_state(True)
            return True
        except Exception as e:
            logger.error(f"Play album at position failed: {e}")
            return False

    def format_playback_info(self, info: Dict) -> str:
        """Format playback info into a human-readable string"""
        if not info:
            return "No music is currently playing."
        
        status = "playing" if info["is_playing"] else "paused"
        
        # Convert milliseconds to minutes:seconds
        duration_sec = info["duration_ms"] // 1000
        progress_sec = info["progress_ms"] // 1000
        remaining_sec = info["remaining_ms"] // 1000
        
        duration_str = f"{duration_sec // 60}:{duration_sec % 60:02d}"
        progress_str = f"{progress_sec // 60}:{progress_sec % 60:02d}"
        remaining_str = f"{remaining_sec // 60}:{remaining_sec % 60:02d}"
        
        return (
            f"Currently {status}: '{info['track_name']}' by {info['artists']} "
            f"from the album '{info['album']}'. "
            f"Progress: {progress_str} of {duration_str} ({info['percent_played']}% complete, "
            f"{remaining_str} remaining)."
        )
