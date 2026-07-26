import pytz
import shutil
import socket
import psutil
import subprocess
import threading
import time
import re
import requests
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SystemInfo:
    """Gather system diagnostics and weather information"""

    def __init__(self, config):
        self.config = config
        self.timezone = pytz.timezone(config.timezone)
        self.gps = None

        # Initialize GPS if enabled in config
        if config.gps.get('enabled', False):
            try:
                from gps_controller import GPSController
                self.gps = GPSController(config)
                if self.gps.start():
                    logger.info("GPS controller started successfully")
                else:
                    # Every start is a cold boot (car cuts Pi power), so gpsd
                    # often isn't accepting connections yet and /dev/ttyACM0
                    # may not have re-enumerated. start() never launches the
                    # reader thread when connect() fails, which also makes
                    # GPSController._reconnect unreachable — so GPS would stay
                    # dead for the whole drive. Retry start() in the background
                    # and KEEP the controller (not None): has_fix() returns
                    # False and get_location() returns None until it connects,
                    # which is exactly what callers already handle.
                    logger.warning("GPS controller failed to start — retrying in background")
                    self._start_gps_retry_thread()
            except ImportError:
                logger.warning("GPS module not found, continuing without GPS")
                self.gps = None
            except Exception as e:
                logger.warning(f"GPS initialization failed: {e}")
                self.gps = None
    
    def _start_gps_retry_thread(self, interval_seconds: float = 15.0, max_attempts: int = 40):
        """Retry GPSController.start() in the background after a failed connect.

        gpsd and the USB GPS can take anywhere from seconds to minutes to come
        up after the car powers the Pi. Once start() succeeds the controller's
        own _gps_loop/_reconnect handles everything from there, so this stops.
        Bounded (~10 minutes) so a genuinely absent GPS doesn't retry forever.
        """
        def retry():
            for attempt in range(1, max_attempts + 1):
                time.sleep(interval_seconds)
                if self.gps is None or self.gps.running:
                    return
                try:
                    if self.gps.start():
                        logger.info(f"GPS controller started on retry {attempt}")
                        return
                except Exception as e:
                    logger.debug(f"GPS retry {attempt} failed: {e}")
            logger.warning("GPS controller still unavailable after retries")

        threading.Thread(target=retry, daemon=True, name="gps-retry").start()

    def get_time_instruction(self) -> str:
        """Get current time formatted for assistant"""
        now = datetime.now(self.timezone)
        return f"The current date and time is {now.strftime('%Y-%m-%d %I:%M %p %Z')}."
    
    def get_detailed_weather(self, location: Optional[str] = None, query_type: str = "current") -> str:
        """
        Get detailed weather information with better formatting

        Args:
            location: City name (optional, uses GPS or default if None)
            query_type: Type of weather query (current, forecast, temperature, conditions)

        Returns:
            Formatted weather string for GPT to use
        """
        api_key = self.config.weather['api_key']

        try:
            # Current weather
            if query_type in ["current", "temperature", "conditions"]:
                # Build URL - use GPS if available and no location override
                url = self._build_weather_url(api_key, use_gps=(location is None), location_override=location)
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                temp = int(data["main"]["temp"])
                feels_like = int(data["main"]["feels_like"])
                temp_min = int(data["main"]["temp_min"])
                temp_max = int(data["main"]["temp_max"])
                humidity = data["main"]["humidity"]
                condition = data["weather"][0]["description"]
                wind_speed = int(data["wind"]["speed"])
                wind_direction = self._degrees_to_direction(data["wind"].get("deg", 0))
                pressure = data["main"]["pressure"]
                visibility_miles = data.get("visibility", 0) / 1609.34
                
                # Check for precipitation
                rain_1h = data.get("rain", {}).get("1h", 0)
                snow_1h = data.get("snow", {}).get("1h", 0)
                
                # Format based on query type
                if query_type == "temperature":
                    return (
                        f"Temperature in {data['name']}: {temp} degrees Fahrenheit, "
                        f"feels like {feels_like} degrees. "
                        f"High of {temp_max}, low of {temp_min}."
                    )
                
                elif query_type == "conditions":
                    precip_text = ""
                    if rain_1h > 0:
                        precip_text = f" Currently raining with {rain_1h:.1f}mm in the last hour."
                    elif snow_1h > 0:
                        precip_text = f" Currently snowing with {snow_1h:.1f}mm in the last hour."
                    
                    return (
                        f"Current conditions in {data['name']}: {condition}. "
                        f"Temperature {temp} degrees, humidity {humidity}%. "
                        f"Wind {wind_speed} mph from the {wind_direction}.{precip_text}"
                    )
                
                else:  # current - comprehensive
                    precip_text = ""
                    if rain_1h > 0:
                        precip_text = f"Rain: {rain_1h:.1f}mm in last hour. "
                    elif snow_1h > 0:
                        precip_text = f"Snow: {snow_1h:.1f}mm in last hour. "
                    
                    return (
                        f"Weather in {data['name']}: {condition}. "
                        f"Temperature: {temp}°F (feels like {feels_like}°F). "
                        f"High: {temp_max}°F, Low: {temp_min}°F. "
                        f"Humidity: {humidity}%. "
                        f"Wind: {wind_speed} mph {wind_direction}. "
                        f"{precip_text}"
                        f"Visibility: {visibility_miles:.1f} miles."
                    )
            
            # Forecast
            elif query_type == "forecast":
                # Build forecast URL with GPS or location
                if location is None and self.gps and self.gps.has_fix():
                    gps_location = self.gps.get_location()
                    if gps_location:
                        lat, lon = gps_location
                        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=imperial&cnt=8"
                    else:
                        query_location = f"{self.config.weather['city']},{self.config.weather['state']},{self.config.weather['country']}"
                        url = f"https://api.openweathermap.org/data/2.5/forecast?q={query_location}&appid={api_key}&units=imperial&cnt=8"
                else:
                    query_location = location or f"{self.config.weather['city']},{self.config.weather['state']},{self.config.weather['country']}"
                    url = f"https://api.openweathermap.org/data/2.5/forecast?q={query_location}&appid={api_key}&units=imperial&cnt=8"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                forecasts = []
                for item in data["list"][:4]:  # Next 12 hours (3-hour intervals)
                    dt = datetime.fromtimestamp(item["dt"], tz=self.timezone)
                    temp = int(item["main"]["temp"])
                    condition = item["weather"][0]["description"]
                    pop = int(item.get("pop", 0) * 100)  # Probability of precipitation
                    
                    time_str = dt.strftime("%I %p")
                    forecast_str = f"{time_str}: {temp}°F, {condition}"
                    if pop > 20:
                        forecast_str += f", {pop}% chance of rain"
                    forecasts.append(forecast_str)
                
                return f"Forecast for {data['city']['name']}: " + " | ".join(forecasts)
        
        except requests.RequestException as e:
            logger.error(f"Weather API error: {e}")
            location_desc = location or "current location"
            return f"Could not retrieve weather for {location_desc}."
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing weather data: {e}")
            return "Error parsing weather information."
    
    def _build_weather_url(self, api_key: str, use_gps: bool = True, location_override: Optional[str] = None) -> str:
        """
        Build weather API URL using GPS coordinates or city name

        Args:
            api_key: OpenWeatherMap API key
            use_gps: Whether to try GPS coordinates first
            location_override: Optional location string to use instead of config

        Returns:
            Weather API URL
        """
        base_url = "https://api.openweathermap.org/data/2.5/weather"

        # Try GPS coordinates first if enabled and available
        if use_gps and self.gps and self.gps.has_fix():
            gps_location = self.gps.get_location()
            if gps_location:
                lat, lon = gps_location
                # Validate coordinates - reject (0.0, 0.0) or invalid ranges
                if abs(lat) > 0.001 and abs(lon) > 0.001 and -90 <= lat <= 90 and -180 <= lon <= 180:
                    logger.info(f"Using GPS coordinates for weather: {lat:.4f}, {lon:.4f}")
                    return f"{base_url}?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
                else:
                    logger.warning(f"GPS coordinates invalid ({lat:.4f}, {lon:.4f}), using configured location")

        # Fall back to location string
        if location_override:
            query_location = location_override
        else:
            city = self.config.weather['city']
            state = self.config.weather['state']
            country = self.config.weather['country']
            query_location = f"{city},{state},{country}"

        logger.info(f"Using configured location for weather: {query_location}")

        logger.debug(f"Using location string for weather: {query_location}")
        return f"{base_url}?q={query_location}&appid={api_key}&units=imperial"

    def _degrees_to_direction(self, degrees: float) -> str:
        """Convert wind degrees to cardinal direction"""
        directions = ["north", "northeast", "east", "southeast",
                     "south", "southwest", "west", "northwest"]
        idx = int((degrees + 22.5) / 45) % 8
        return directions[idx]
    
    def get_system_diagnostics(self) -> str:
        """Get Raspberry Pi system diagnostics"""
        # CPU temperature
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                cpu_temp = f"{int(f.read()) / 1000:.1f}°C"
        except Exception:
            cpu_temp = "Unavailable"
        
        # Disk usage
        total, used, free = shutil.disk_usage("/")
        total_gb = total // (2**30)
        free_gb = free // (2**30)
        
        # RAM
        try:
            mem = psutil.virtual_memory()
            total_mem_mb = mem.total // (2**20)
            used_mem_mb = (mem.total - mem.available) // (2**20)
            mem_usage = f"{used_mem_mb} MB used / {total_mem_mb} MB"
        except Exception:
            mem_usage = "Unavailable"
        
        # IP address
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip_address = s.getsockname()[0]
        except Exception:
            ip_address = "Unavailable"
        
        # WiFi SSID
        ssid = self._get_wifi_ssid()
        
        # Signal strength
        signal_strength = self._get_signal_strength()
        
        return (
            "System Diagnostics:\n"
            "CPU: Quad Core Broadcom BCM2712 ARM Cortex A76 processor at 2.4GHz\n"
            f"CPU Temperature (Celsius): {cpu_temp}\n"
            f"Memory Usage: {mem_usage}\n"
            f"Disk Space: {total_gb} GB total, {free_gb} GB free\n"
            f"IP Address: {ip_address}\n"
            f"Wi-Fi Network: {ssid}\n"
            f"Signal Strength: {signal_strength}"
        )
    
    def _get_wifi_ssid(self) -> str:
        """Get current WiFi SSID"""
        try:
            return subprocess.check_output(
                ["/usr/bin/iwgetid", "-r"], text=True
            ).strip() or "Not connected"
        except (FileNotFoundError, subprocess.CalledProcessError):
            try:
                nmcli_output = subprocess.check_output(
                    ["/usr/bin/nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                    text=True
                )
                for line in nmcli_output.strip().split("\n"):
                    if line.startswith("yes:"):
                        return line.split(":")[1]
                return "Not connected"
            except Exception:
                return "Unavailable"
    
    def _get_signal_strength(self) -> str:
        """Get WiFi signal strength"""
        try:
            iwconfig_output = subprocess.check_output(["/usr/sbin/iwconfig"], text=True)
            m = re.search(r"Signal level=(-?\d+) dBm", iwconfig_output)
            if m:
                return m.group(1) + " dBm"
        except FileNotFoundError:
            try:
                nmcli_output = subprocess.check_output(
                    ["/usr/bin/nmcli", "-t", "-f", "active,signal", "dev", "wifi"],
                    text=True
                )
                for line in nmcli_output.strip().split("\n"):
                    if line.startswith("yes:"):
                        strength_percent = int(line.split(":")[1])
                        return f"{-100 + (strength_percent * 0.6):.0f} dBm"
            except Exception:
                pass
        except Exception:
            pass
        return "Unavailable"
