# -*- coding: utf-8 -*-
"""
Utility Functions Module
Contains helper functions for INI parsing, path handling, etc.
"""
import configparser
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# LRU Cache for parsed INI files
class INICache:
    """Simple LRU Cache for parsed INI files"""

    def __init__(self, max_size: int = 100):
        """Initialize cache with maximum size"""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
        logger.info(f"INI cache initialized with max size: {max_size}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get parsed INI data from cache if it exists"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key: str, ini_data: Dict[str, Any]) -> None:
        """Add parsed INI data to cache, possibly evicting oldest entries"""
        # Check if cache is full and needs eviction
        if len(self.cache) >= self.max_size:
            # Find least recently used entry
            oldest_key = min(
                self.access_times.keys(), key=lambda k: self.access_times[k]
            )
            # Remove it
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        # Add new entry
        self.cache[key] = ini_data
        self.access_times[key] = time.time()

    def clear(self) -> None:
        """Clear the INI cache"""
        self.cache.clear()
        self.access_times.clear()
        logger.info("INI cache cleared")


# Create global INI cache
ini_cache = INICache()


def parse_ini_file(ini_path: str) -> Dict[str, Any]:
    """
    Parse an INI file and extract settings

    Args:
        ini_path: Path to INI file

    Returns:
        Dictionary with settings from [MatchSettings] section

    Raises:
        FileNotFoundError: If the INI file does not exist
        ValueError: If the INI file has missing required sections or invalid data
    """
    # Check cache first
    cached_data = ini_cache.get(ini_path)
    if cached_data:
        return cached_data

    # Strip potential whitespace and newlines from the path
    ini_path = ini_path.strip()

    # Check if file exists
    if not os.path.exists(ini_path):
        logger.error(f"INI file not found: {ini_path}")
        raise FileNotFoundError(f"INI file not found: {ini_path}")

    try:
        # Create parser and read the file
        config = configparser.ConfigParser()
        config.read(ini_path, encoding="utf-8")

        # Check if [MatchSettings] section exists
        if "MatchSettings" not in config:
            logger.error(f"Missing [MatchSettings] section in {ini_path}")
            raise ValueError(f"Missing [MatchSettings] section in {ini_path}")

        # Extract settings from [MatchSettings] section
        settings = {}
        for key, value in config["MatchSettings"].items():
            # Convert empty strings to None
            if value.strip() == "":
                settings[key] = None
            else:
                settings[key] = value

        # Add derived template path (replace .ini with .jpg)
        ini_basename = os.path.basename(ini_path)
        template_name = os.path.splitext(ini_basename)[0] + ".jpg"
        template_dir = os.path.dirname(ini_path)
        template_path = os.path.join(template_dir, template_name)

        settings["template_path"] = template_path
        settings["ini_path"] = ini_path

        # Cache the parsed data
        ini_cache.put(ini_path, settings)

        return settings

    except configparser.Error as e:
        logger.error(f"Error parsing INI file {ini_path}: {e}")
        raise ValueError(f"Error parsing INI file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing INI file {ini_path}: {e}")
        raise


def parse_query_params(params: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse and convert query parameters from API request

    Args:
        params: Raw query parameters from request

    Returns:
        Dictionary with properly typed parameters
    """
    result = {}

    # Process each parameter
    for key, value in params.items():
        # Skip empty values
        if value is None or value.strip() == "":
            continue

        # Convert boolean values
        if value.lower() in ("true", "yes", "1"):
            result[key] = True
        elif value.lower() in ("false", "no", "0"):
            result[key] = False
        else:
            # Try to convert to number if possible
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                # Keep as string if not a number
                result[key] = value

    return result


def get_match_range_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize match range parameters

    Args:
        params: Parameters dictionary

    Returns:
        Dictionary with normalized match range parameters
    """
    result = {}

    # Handle both x1/y1/x2/y2 (from /search) and match_range_x1/etc (from INI)
    if "x1" in params:
        result["match_range_x1"] = params["x1"]
    elif "match_range_x1" in params:
        result["match_range_x1"] = params["match_range_x1"]

    if "y1" in params:
        result["match_range_y1"] = params["y1"]
    elif "match_range_y1" in params:
        result["match_range_y1"] = params["match_range_y1"]

    if "x2" in params:
        result["match_range_x2"] = params["x2"]
    elif "match_range_x2" in params:
        result["match_range_x2"] = params["match_range_x2"]

    if "y2" in params:
        result["match_range_y2"] = params["y2"]
    elif "match_range_y2" in params:
        result["match_range_y2"] = params["match_range_y2"]

    # Handle offset parameters
    if "offsetx" in params:
        result["offset_x"] = params["offsetx"]
    elif "offset_x" in params:
        result["offset_x"] = params["offset_x"]

    if "offsety" in params:
        result["offset_y"] = params["offsety"]
    elif "offset_y" in params:
        result["offset_y"] = params["offset_y"]

    return result


def validate_filter_type(filter_type: str) -> str:
    """
    Validate filter type and return normalized value

    Args:
        filter_type: Filter type to validate

    Returns:
        Normalized filter type

    Raises:
        ValueError: If filter type is invalid
    """
    valid_types = ["none", "canny"]
    normalized = filter_type.lower() if filter_type else "none"

    if normalized not in valid_types:
        logger.warning(f"Invalid filter type: {filter_type}. Using 'none'.")
        return "none"

    return normalized


def validate_match_method(match_method: str) -> str:
    """
    Validate match method and return normalized value

    Args:
        match_method: Match method to validate

    Returns:
        Normalized match method

    Raises:
        ValueError: If match method is invalid
    """
    valid_methods = ["ccoeff_normed", "sqdiff_normed", "ccorr_normed"]
    normalized = match_method.lower() if match_method else "ccoeff_normed"

    if normalized not in valid_methods:
        logger.warning(f"Invalid match method: {match_method}. Using 'ccoeff_normed'.")
        return "ccoeff_normed"

    return normalized


def clean_debug_images(debug_dir: str, max_files: int = 100) -> None:
    """
    Clean up debug images directory to keep it under specified size

    Args:
        debug_dir: Path to debug images directory
        max_files: Maximum number of files to keep
    """
    if not os.path.exists(debug_dir):
        return

    try:
        # List all files in the directory with full paths
        files = [
            os.path.join(debug_dir, f)
            for f in os.listdir(debug_dir)
            if os.path.isfile(os.path.join(debug_dir, f))
        ]

        # Check if we need to clean up
        if len(files) <= max_files:
            return

        # Sort files by modification time (oldest first)
        files.sort(key=os.path.getmtime)

        # Delete oldest files to get down to max_files
        files_to_delete = files[: (len(files) - max_files)]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                logger.debug(f"Removed old debug file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove debug file {file_path}: {e}")

    except Exception as e:
        logger.error(f"Error cleaning debug directory {debug_dir}: {e}")


def save_debug_frame(
    frame, debug_dir: str, prefix: str = "frame", max_files: int = 100
) -> Optional[str]:
    """
    Save a debug frame to disk

    Args:
        frame: Frame to save (numpy array)
        debug_dir: Directory to save frame in
        prefix: Filename prefix
        max_files: Maximum number of files to keep

    Returns:
        Path to saved file or None if saving failed
    """
    import cv2

    if frame is None:
        logger.warning("Cannot save debug frame: frame is None")
        return None

    # Ensure debug directory exists
    os.makedirs(debug_dir, exist_ok=True)

    # Clean up old files if needed
    clean_debug_images(debug_dir, max_files)

    try:
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        filename = f"{prefix}_{timestamp}_{ms:03d}.jpg"
        filepath = os.path.join(debug_dir, filename)

        # Save the frame
        cv2.imwrite(filepath, frame)
        logger.debug(f"Saved debug frame to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Failed to save debug frame: {e}")
        return None
