import copy
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for VIVIAN"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to config.yaml file (default: "config.yaml")
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
        logger.info(f"Configuration loaded from {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create config.yaml in the same directory as main.py"
            )
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        
        if not config:
            raise ValueError("Config file is empty")
        
        # Override with environment variables if present
        self._apply_env_overrides(config)
        
        # Validate required sections
        self._validate_config(config)
        
        return config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> None:
        """
        Override config values with environment variables.
        This allows secrets to be stored in env vars instead of config.yaml
        """
        env_mappings = {
            'PORCUPINE_ACCESS_KEY': ('porcupine', 'access_key'),
            'SPOTIFY_CLIENT_ID': ('spotify', 'client_id'),
            'SPOTIFY_CLIENT_SECRET': ('spotify', 'client_secret'),
            'SPOTIFY_REFRESH_TOKEN': ('spotify', 'refresh_token'),
            'ANTHROPIC_API_KEY': ('anthropic', 'api_key'),
            'OPENWEATHER_API_KEY': ('weather', 'api_key'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                if section in config and isinstance(config[section], dict):
                    config[section][key] = os.environ[env_var]
                    logger.info(f"Overriding {section}.{key} from environment variable {env_var}")
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate that required configuration sections exist"""
        required_sections = [
            'porcupine',
            'spotify',
            'anthropic',
            'tts',
            'audio',
            'whisper',
            'files',
            'hardware',
            'weather',
            'vehicle'
        ]
        
        missing_sections = [s for s in required_sections if s not in config]
        if missing_sections:
            raise ValueError(
                f"Missing required sections in config.yaml: {', '.join(missing_sections)}"
            )
        
        logger.debug("All required configuration sections present")
    
    def get(self, *keys, default=None):
        """
        Get nested configuration value.
        
        Args:
            *keys: Variable number of keys for nested access
            default: Default value if key path doesn't exist
        
        Returns:
            Configuration value or default
        
        Example:
            config.get('spotify', 'client_id')
            config.get('audio', 'sample_rate', default=48000)
        """
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {'.'.join(keys)}")
    
    @property
    def porcupine(self) -> Dict[str, Any]:
        """Porcupine wake word configuration"""
        return self._config['porcupine']
    
    @property
    def spotify(self) -> Dict[str, Any]:
        """Spotify integration configuration"""
        return self._config['spotify']
    
    @property
    def anthropic(self) -> Dict[str, Any]:
        """Anthropic API configuration"""
        return self._config['anthropic']

    @property
    def tts(self) -> Dict[str, Any]:
        """Local TTS (Kokoro) configuration"""
        return self._config['tts']
    
    @property
    def audio(self) -> Dict[str, Any]:
        """Audio recording configuration"""
        return self._config['audio']
    
    @property
    def whisper(self) -> Dict[str, Any]:
        """Whisper transcription configuration"""
        return self._config['whisper']
    
    @property
    def files(self) -> Dict[str, Any]:
        """File paths configuration"""
        return self._config['files']
    
    @property
    def hardware(self) -> Dict[str, Any]:
        """Hardware (GPIO) configuration"""
        return self._config['hardware']
    
    @property
    def weather(self) -> Dict[str, Any]:
        """Weather API configuration"""
        return self._config['weather']
    
    @property
    def vehicle(self) -> Dict[str, Any]:
        """Vehicle specifications"""
        return self._config['vehicle']

    @property
    def timezone(self) -> str:
        """IANA timezone name shared across modules (default US/Eastern)"""
        return self._config.get('timezone', 'US/Eastern')

    @property
    def gps(self) -> Dict[str, Any]:
        """GPS configuration (optional)"""
        return self._config.get('gps', {'enabled': False})

    @property
    def sentry(self) -> Dict[str, Any]:
        """Sentry mode configuration (optional)"""
        return self._config.get('sentry', {})

    @property
    def glass(self) -> Dict[str, Any]:
        """Glass HUD configuration (optional)"""
        return self._config.get('glass', {'enabled': False})

    @property
    def dashboard(self) -> Dict[str, Any]:
        """Dashboard configuration (optional)"""
        return self._config.get('dashboard', {'enabled': False})

    def reload(self) -> None:
        """Reload configuration from file"""
        logger.info("Reloading configuration...")
        self._config = self._load_config()
        logger.info("Configuration reloaded")
    
    def __repr__(self) -> str:
        """String representation of config (hides sensitive values)"""
        # Deep copy: a shallow .copy() shares the nested section dicts, so
        # masking here would overwrite the REAL keys in the live config.
        safe_config = copy.deepcopy(self._config)
        
        # Hide sensitive values
        sensitive_keys = ['access_key', 'client_id', 'client_secret', 'refresh_token', 'api_key']
        
        for section in safe_config:
            if isinstance(safe_config[section], dict):
                for key in sensitive_keys:
                    if key in safe_config[section]:
                        safe_config[section][key] = "***HIDDEN***"
        
        return f"Config({safe_config})"
