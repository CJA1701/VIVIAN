from gps import gps, WATCH_ENABLE, WATCH_NEWSTYLE
import math
import threading
import time
import json
import os
import logging
from typing import Optional, Dict, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Shared GPS data file - other modules (OLED, etc.) read from this
GPS_SHARED_FILE = "/tmp/vivian_gps.json"


class GPSController:
    """
    GPS controller for VK-162 USB GPS adapter
    Provides location data, speed, altitude, and satellite information
    """
    
    def __init__(self, config=None):
        """
        Initialize GPS controller
        
        Args:
            config: Optional Config object (not used currently, for future expansion)
        """
        self.config = config
        self.gpsd = None
        self.running = False
        self.thread = None
        
        # Current GPS data
        self._lock = threading.Lock()
        self._latitude = None
        self._longitude = None
        self._altitude = None
        self._speed = None  # Speed in meters per second
        self._track = None  # Track angle in degrees
        self._climb = None  # Climb rate in meters per second
        self._time = None   # GPS time
        self._mode = 0      # GPS fix mode (0/1=no fix, 2=2D, 3=3D)
        self._satellites_used = 0
        self._satellites_visible = 0
        self._last_update = None
        self._last_file_write = 0  # Timestamp of last shared file write
        self._consecutive_errors = 0  # Read errors since last good report

        # Calculated speed from position deltas (fallback for GPS modules
        # that report speed as 0, like the VK-162)
        self._calc_speed = None  # meters per second
        self._prev_lat = None
        self._prev_lon = None
        self._prev_time = None

        logger.info("GPS controller initialized")
    
    def connect(self):
        """Connect to GPS daemon"""
        try:
            self.gpsd = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)
            logger.info("Connected to GPS daemon")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to GPS daemon: {e}")
            logger.error("Make sure gpsd is running: sudo gpsd /dev/ttyACM0 -F /var/run/gpsd.sock -n")
            return False
    
    def start(self):
        """Start GPS data collection in background thread"""
        if self.running:
            logger.warning("GPS controller already running")
            return False
        
        if not self.gpsd:
            if not self.connect():
                return False
        
        self.running = True
        self.thread = threading.Thread(target=self._gps_loop, daemon=True)
        self.thread.start()
        logger.info("GPS data collection started")
        return True
    
    def stop(self):
        """Stop GPS data collection"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.gpsd:
            try:
                self.gpsd.close()
            except:
                pass
        
        logger.info("GPS data collection stopped")
    
    def _gps_loop(self):
        """Background loop to continuously read GPS data"""
        while self.running:
            try:
                report = self.gpsd.next()
                
                # Process TPV (Time Position Velocity) reports
                if report.get('class') == 'TPV':
                    with self._lock:
                        self._mode = report.get('mode', 0)

                        lat = report.get('lat')
                        lon = report.get('lon')

                        # ECEF fallback: VK-162 often reports lat/lon as 0.0
                        # but provides valid ECEF coordinates
                        if (lat is None or (lat == 0.0 and lon == 0.0)) and 'ecefx' in report:
                            ecef_lat, ecef_lon, ecef_alt = self._ecef_to_geodetic(
                                report['ecefx'], report['ecefy'], report['ecefz']
                            )
                            if ecef_lat is not None:
                                lat = ecef_lat
                                lon = ecef_lon
                                if self._altitude is None or self._altitude == 0.0:
                                    self._altitude = ecef_alt
                                logger.debug(f"Using ECEF fallback: lat={lat:.6f}, lon={lon:.6f}")

                        if lat is not None:
                            self._latitude = lat
                        if lon is not None:
                            self._longitude = lon
                        if 'alt' in report or 'altHAE' in report:
                            gpsd_alt = report.get('altHAE', report.get('alt'))
                            # Don't overwrite valid ECEF altitude with gpsd's 0.0
                            if gpsd_alt and abs(gpsd_alt) > 0.1:
                                self._altitude = gpsd_alt
                        if 'speed' in report:
                            self._speed = report['speed']
                        if 'track' in report:
                            self._track = report['track']
                        if 'climb' in report:
                            self._climb = report['climb']
                        if 'time' in report:
                            self._time = report['time']

                        # Calculate speed from position deltas as fallback
                        # (VK-162 reports speed as 0 even when moving)
                        if lat is not None and lon is not None and abs(lat) > 0.01:
                            # monotonic: this Pi has no RTC battery, so NTP steps
                            # the wall clock at every boot. A forward step
                            # inflated dt while still passing the 0.2-5.0s gate
                            # (silently under-reporting speed).
                            now = time.monotonic()
                            if self._prev_lat is not None and self._prev_time is not None:
                                dt = now - self._prev_time
                                if 0.2 < dt < 5.0:
                                    dist = self._calculate_distance(
                                        self._prev_lat, self._prev_lon, lat, lon
                                    )
                                    self._calc_speed = dist / dt
                            self._prev_lat = lat
                            self._prev_lon = lon
                            self._prev_time = now

                        self._last_update = datetime.now()

                        logger.debug(f"GPS update: lat={self._latitude}, lon={self._longitude}, "
                                   f"mode={self._mode}, speed={self._speed}")
                
                # Process SKY (Satellite) reports
                elif report.get('class') == 'SKY':
                    with self._lock:
                        sats = report.get('satellites', [])
                        self._satellites_visible = len(sats)
                        self._satellites_used = sum(1 for s in sats if s.get('used', False))

                # Write to shared file periodically (every 0.25s)
                # monotonic: a BACKWARD clock step made this difference
                # negative, so the shared file stopped being written until the
                # wall clock caught up — the CRT and Glass froze their speed and
                # position for the rest of the drive, with nothing logged.
                if time.monotonic() - self._last_file_write >= 0.25:
                    self._write_shared_file()
                    self._last_file_write = time.monotonic()

            except StopIteration:
                # gpsd closed the session (restart, USB re-enumeration, ...)
                # — reconnect instead of leaving GPS dead until reboot.
                logger.warning("GPS session ended — reconnecting to gpsd")
                self._reconnect()
            except KeyError as e:
                # Some reports don't have all fields, this is normal
                logger.debug(f"Missing field in GPS report: {e}")
            except Exception as e:
                logger.error(f"Error reading GPS data: {e}")
                self._consecutive_errors += 1
                if self._consecutive_errors >= 10:
                    logger.warning("Repeated GPS read errors — reconnecting to gpsd")
                    self._reconnect()
                else:
                    time.sleep(1)
            else:
                self._consecutive_errors = 0

    def _reconnect(self, delay: float = 5.0):
        """Close the gpsd session and reconnect with backoff until it works."""
        try:
            if self.gpsd:
                self.gpsd.close()
        except Exception:
            pass
        self.gpsd = None
        self._consecutive_errors = 0

        while self.running:
            time.sleep(delay)
            if self.connect():
                logger.info("Reconnected to gpsd")
                return
            delay = min(delay * 2, 60.0)  # back off up to 1 minute

    @staticmethod
    def _ecef_to_geodetic(x, y, z):
        """Convert ECEF (Earth-Centered Earth-Fixed) coordinates to geodetic (lat, lon, alt).

        Uses iterative Bowring method on WGS84 ellipsoid.

        Args:
            x, y, z: ECEF coordinates in meters

        Returns:
            Tuple of (latitude_degrees, longitude_degrees, altitude_meters) or (None, None, None)
        """
        try:
            # WGS84 constants
            a = 6378137.0           # Semi-major axis (meters)
            f = 1 / 298.257223563   # Flattening
            b = a * (1 - f)         # Semi-minor axis
            e2 = 1 - (b * b) / (a * a)  # First eccentricity squared
            ep2 = (a * a) / (b * b) - 1  # Second eccentricity squared

            lon = math.atan2(y, x)
            p = math.sqrt(x * x + y * y)

            # Initial latitude estimate (Bowring)
            theta = math.atan2(z * a, p * b)
            lat = math.atan2(
                z + ep2 * b * math.sin(theta) ** 3,
                p - e2 * a * math.cos(theta) ** 3
            )

            # Iterate for precision
            for _ in range(5):
                sin_lat = math.sin(lat)
                N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
                lat = math.atan2(z + e2 * N * sin_lat, p)

            sin_lat = math.sin(lat)
            N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
            alt = p / math.cos(lat) - N

            lat_deg = math.degrees(lat)
            lon_deg = math.degrees(lon)

            # Sanity check
            if -90 <= lat_deg <= 90 and -180 <= lon_deg <= 180:
                return lat_deg, lon_deg, alt
            return None, None, None
        except Exception:
            return None, None, None

    def _write_shared_file(self):
        """Write GPS data to shared file for other modules (OLED, etc.)"""
        try:
            with self._lock:
                # Use calculated speed if gpsd reports 0
                effective_speed = self._speed
                if (effective_speed is None or effective_speed == 0.0) and self._calc_speed is not None:
                    effective_speed = self._calc_speed
                data = {
                    'lat': self._latitude,
                    'lon': self._longitude,
                    'alt': self._altitude,
                    'speed_mps': effective_speed,
                    'speed_mph': effective_speed * 2.23694 if effective_speed is not None else None,
                    'track': self._track,
                    'mode': self._mode,
                    'sats_used': self._satellites_used,
                    'sats_visible': self._satellites_visible,
                    'has_fix': self._mode >= 2 and self._latitude is not None and abs(self._latitude) > 0.01,
                    'timestamp': time.time()
                }
            # Write atomically using temp file + rename
            tmp_file = GPS_SHARED_FILE + '.tmp'
            with open(tmp_file, 'w') as f:
                json.dump(data, f)
            os.replace(tmp_file, GPS_SHARED_FILE)
        except Exception as e:
            logger.debug(f"Failed to write shared GPS file: {e}")

    def get_location(self) -> Optional[Tuple[float, float]]:
        """
        Get current GPS location

        Returns:
            Tuple of (latitude, longitude) or None if no fix or invalid coordinates
        """
        with self._lock:
            if self._mode >= 2 and self._latitude is not None and self._longitude is not None:
                # Validate coordinates - reject (0.0, 0.0) or out of range values
                if abs(self._latitude) > 0.001 and abs(self._longitude) > 0.001:
                    if -90 <= self._latitude <= 90 and -180 <= self._longitude <= 180:
                        return (self._latitude, self._longitude)
            return None
    
    def get_full_data(self) -> Dict:
        """
        Get all available GPS data
        
        Returns:
            Dictionary with all GPS information
        """
        with self._lock:
            speed = self._speed
            if (speed is None or speed == 0.0) and self._calc_speed is not None:
                speed = self._calc_speed
            return {
                'latitude': self._latitude,
                'longitude': self._longitude,
                'altitude': self._altitude,
                'speed_mps': speed,  # meters per second
                'speed_mph': speed * 2.23694 if speed is not None else None,
                'speed_kph': speed * 3.6 if speed is not None else None,
                'track': self._track,
                'climb': self._climb,
                'time': self._time,
                'mode': self._mode,
                'mode_name': self._get_mode_name(self._mode),
                'satellites_used': self._satellites_used,
                'satellites_visible': self._satellites_visible,
                'last_update': self._last_update,
                'has_fix': self._mode >= 2
            }
    
    def get_address(self) -> Optional[str]:
        """
        Get approximate address from coordinates using reverse geocoding
        Note: Requires internet connection and geopy library

        Returns:
            Address string or None
        """
        location = self.get_location()
        if not location:
            return None

        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="VIVIAN")
            location_obj = geolocator.reverse(f"{location[0]}, {location[1]}")
            return location_obj.address if location_obj else None
        except ImportError:
            logger.warning("geopy not installed. Install with: pip3 install geopy")
            return None
        except Exception as e:
            logger.error(f"Reverse geocoding failed: {e}")
            return None

    def get_detailed_address(self) -> Optional[Dict]:
        """
        Get detailed address components from coordinates

        Returns:
            Dictionary with address components or None
            {
                'house_number': str,
                'road': str,
                'neighbourhood': str,
                'city': str,
                'county': str,
                'state': str,
                'postcode': str,
                'country': str,
                'formatted': str (full address string)
            }
        """
        location = self.get_location()
        if not location:
            return None

        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="VIVIAN")
            location_obj = geolocator.reverse(f"{location[0]}, {location[1]}", addressdetails=True)

            if location_obj and location_obj.raw.get('address'):
                address = location_obj.raw['address']
                return {
                    'house_number': address.get('house_number', ''),
                    'road': address.get('road', ''),
                    'neighbourhood': address.get('neighbourhood', ''),
                    'suburb': address.get('suburb', ''),
                    'city': address.get('city') or address.get('town') or address.get('village', ''),
                    'county': address.get('county', ''),
                    'state': address.get('state', ''),
                    'postcode': address.get('postcode', ''),
                    'country': address.get('country', ''),
                    'formatted': location_obj.address
                }

            return None
        except ImportError:
            logger.warning("geopy not installed. Install with: pip3 install geopy")
            return None
        except Exception as e:
            logger.error(f"Detailed address lookup failed: {e}")
            return None

    def find_specific_business(self, business_name: str, radius_meters: int = 16000) -> Optional[List[Dict]]:
        """
        Find a specific business by name using Google Places API (e.g., "McDonald's", "Shell", "Walmart")

        Args:
            business_name: Name of the business to search for
            radius_meters: Search radius in meters (default 16000m = ~10 miles)

        Returns:
            List of matching businesses with distances, sorted by proximity
        """
        location = self.get_location()
        if not location:
            return None

        # Get API key from config
        api_key = None
        if self.config:
            api_key = self.config.gps.get('google_api_key')

        if not api_key:
            logger.error("Google Places API key not configured")
            return None

        try:
            import requests

            lat, lon = location

            # Use Google Places Text Search API
            url = "https://places.googleapis.com/v1/places:searchText"

            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.types"
            }

            payload = {
                "textQuery": business_name,
                "locationBias": {
                    "circle": {
                        "center": {
                            "latitude": lat,
                            "longitude": lon
                        },
                        "radius": radius_meters
                    }
                },
                "maxResultCount": 20
            }

            logger.debug(f"Searching for '{business_name}' via Google Places API")
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            for place in data.get('places', []):
                place_lat = place.get('location', {}).get('latitude')
                place_lon = place.get('location', {}).get('longitude')

                if not place_lat or not place_lon:
                    continue

                # Calculate distance
                distance_meters = self._calculate_distance(lat, lon, place_lat, place_lon)

                # Only include results within radius
                if distance_meters > radius_meters:
                    continue

                name = place.get('displayName', {}).get('text', 'Unnamed')
                address = place.get('formattedAddress', 'Address not available')
                types = place.get('types', [])
                place_type = types[0] if types else 'business'

                results.append({
                    'name': name,
                    'type': place_type,
                    'distance_meters': round(distance_meters, 1),
                    'distance_miles': round(distance_meters / 1609.34, 2),
                    'address': address,
                    'lat': place_lat,
                    'lon': place_lon
                })

            # Sort by distance
            results.sort(key=lambda x: x['distance_meters'])

            return results if results else None

        except ImportError:
            logger.warning("requests library needed for business search")
            return None
        except Exception as e:
            logger.error(f"Business search failed: {e}")
            return None

    def find_nearby_places(self, place_type: str = "amenity", search_query: str = None,
                          radius_meters: int = 16000, limit: int = 10) -> Optional[List[Dict]]:
        """
        Find nearby places of interest using Google Places API

        Args:
            place_type: Category type (used for mapping to Google types)
            search_query: Specific place type (gas_station, restaurant, fast_food, etc.)
            radius_meters: Search radius in meters (default 16000m = ~10 miles)
            limit: Maximum number of results

        Returns:
            List of dictionaries with place information or None
            [{
                'name': str,
                'type': str,
                'distance_meters': float,
                'distance_miles': float,
                'address': str,
                'lat': float,
                'lon': float
            }]
        """
        location = self.get_location()
        if not location:
            return None

        # Get API key from config
        api_key = None
        if self.config:
            api_key = self.config.gps.get('google_api_key')

        if not api_key:
            logger.error("Google Places API key not configured")
            return None

        try:
            import requests

            lat, lon = location

            # Map common queries to Google Place types
            type_mapping = {
                'gas_station': 'gas_station',
                'fuel': 'gas_station',
                'fast_food': 'fast_food_restaurant',
                'restaurant': 'restaurant',
                'cafe': 'cafe',
                'coffee': 'cafe',
                'hotel': 'lodging',
                'atm': 'atm',
                'bank': 'bank',
                'pharmacy': 'pharmacy',
                'hospital': 'hospital',
                'police': 'police',
                'parking': 'parking',
                'charging_station': 'electric_vehicle_charging_station'
            }

            # Get the Google type
            google_type = type_mapping.get(search_query, search_query)

            # Use Google Places Nearby Search API
            url = "https://places.googleapis.com/v1/places:searchNearby"

            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.types"
            }

            payload = {
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat,
                            "longitude": lon
                        },
                        "radius": radius_meters
                    }
                },
                "maxResultCount": limit
            }

            # Add type filter if specified
            if google_type:
                payload["includedTypes"] = [google_type]

            logger.debug(f"Searching nearby places via Google Places API: {google_type}")
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            for place in data.get('places', []):
                place_lat = place.get('location', {}).get('latitude')
                place_lon = place.get('location', {}).get('longitude')

                if not place_lat or not place_lon:
                    continue

                # Calculate distance
                distance_meters = self._calculate_distance(lat, lon, place_lat, place_lon)

                name = place.get('displayName', {}).get('text', 'Unnamed')
                address = place.get('formattedAddress', 'Address not available')
                types = place.get('types', [])
                place_type_value = search_query or (types[0] if types else 'unknown')

                results.append({
                    'name': name,
                    'type': place_type_value,
                    'distance_meters': round(distance_meters, 1),
                    'distance_miles': round(distance_meters / 1609.34, 2),
                    'address': address,
                    'lat': place_lat,
                    'lon': place_lon
                })

            # Sort by distance
            results.sort(key=lambda x: x['distance_meters'])

            return results[:limit] if results else None

        except ImportError:
            logger.warning("requests library needed for POI search")
            return None
        except Exception as e:
            logger.error(f"POI search failed: {e}")
            return None

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates using Haversine formula

        Returns:
            Distance in meters
        """
        import math

        R = 6371000  # Earth's radius in meters

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        return R * c
    
    def get_city_state(self) -> Optional[Tuple[str, str]]:
        """
        Get city and state from GPS coordinates
        
        Returns:
            Tuple of (city, state) or None
        """
        location = self.get_location()
        if not location:
            return None
        
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="VIVIAN")
            location_obj = geolocator.reverse(f"{location[0]}, {location[1]}")
            
            if location_obj and location_obj.raw.get('address'):
                address = location_obj.raw['address']
                city = address.get('city') or address.get('town') or address.get('village')
                state = address.get('state')
                return (city, state) if city and state else None
            
            return None
        except ImportError:
            logger.warning("geopy not installed. Install with: pip3 install geopy")
            return None
        except Exception as e:
            logger.error(f"Failed to get city/state: {e}")
            return None
    
    def _get_mode_name(self, mode: int) -> str:
        """Convert GPS mode number to name"""
        mode_names = {
            0: 'No Fix',
            1: 'No Fix',
            2: '2D Fix',
            3: '3D Fix'
        }
        return mode_names.get(mode, 'Unknown')
    
    def has_fix(self) -> bool:
        """Check if GPS has a valid fix"""
        with self._lock:
            return self._mode >= 2
    
    def wait_for_fix(self, timeout: float = 60.0) -> bool:
        """
        Wait for GPS to acquire a fix
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if fix acquired, False if timeout
        """
        start_time = time.monotonic()
        logger.info("Waiting for GPS fix...")
        
        while time.monotonic() - start_time < timeout:
            if self.has_fix():
                logger.info(f"GPS fix acquired after {time.monotonic() - start_time:.1f} seconds")
                return True
            time.sleep(0.5)
        
        logger.warning(f"GPS fix timeout after {timeout} seconds")
        return False
    
    def get_speed_mph(self) -> Optional[float]:
        """Get current speed in miles per hour"""
        with self._lock:
            speed = self._speed
            if (speed is None or speed == 0.0) and self._calc_speed is not None:
                speed = self._calc_speed
            return speed * 2.23694 if speed is not None else None
    
    def get_formatted_location(self) -> str:
        """Get human-readable location string"""
        location = self.get_location()
        if not location:
            return "GPS: No fix"
        
        lat, lon = location
        data = self.get_full_data()
        
        # Format coordinates
        lat_dir = 'N' if lat >= 0 else 'S'
        lon_dir = 'E' if lon >= 0 else 'W'
        
        result = f"{abs(lat):.6f}°{lat_dir}, {abs(lon):.6f}°{lon_dir}"
        
        # Add altitude if available
        if data['altitude'] is not None:
            result += f", {data['altitude']:.1f}m altitude"
        
        # Add speed if moving
        if data['speed_mph'] is not None and data['speed_mph'] > 1.0:
            result += f", {data['speed_mph']:.1f} mph"
        
        return result
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
