# -*- coding: utf-8 -*-
"""
Utility Functions Module
Contains helper functions for INI parsing, path handling, etc.
"""
import configparser
import logging
import os
import socket
import threading  # 将 threading 导入移动到此处
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class INICache:
    """Simple LRU Cache for parsed INI files"""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}  # Stores access time for LRU
        self._lock = threading.Lock()  # Added lock for thread safety
        logger.info(f"INI cache initialized with max size: {max_size}")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()  # Update access time
                # Move to end to mark as most recently used (if using OrderedDict for LRU)
                # For simple dict + access_times, just updating time is enough for eviction logic
                return self.cache[key]
            return None

    def put(self, key: str, ini_data: Dict[str, Any]) -> None:
        with self._lock:
            if (
                len(self.cache) >= self.max_size and key not in self.cache
            ):  # Evict only if full and new key
                # Find least recently used entry
                try:
                    oldest_key = min(self.access_times, key=self.access_times.get)
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
                    logger.debug(f"INI Cache: Evicted '{oldest_key}'")
                except (
                    ValueError,
                    TypeError,
                ) as e:  # Handle empty access_times or other issues
                    logger.warning(
                        f"INI Cache: Could not determine oldest key for eviction: {e}"
                    )

            self.cache[key] = ini_data
            self.access_times[key] = time.time()
            logger.debug(f"INI Cache: Stored '{key}'")

    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            logger.info("INI cache cleared")


# Global INI cache instance
# Consider initializing this based on server config if cache size needs to be configurable
ini_cache = INICache()


def parse_ini_file(ini_path: str) -> Dict[str, Any]:
    """
    Parses an INI file, expecting a [MatchSettings] section.
    Values are converted to expected types. Paths are resolved.

    Args:
        ini_path: Absolute path to the INI file.

    Returns:
        A dictionary containing settings from [MatchSettings].

    Raises:
        FileNotFoundError: If the INI file does not exist.
        ValueError: If the INI file is malformed, missing [MatchSettings],
                    or contains invalid values for expected types.
    """
    if not os.path.isabs(ini_path):
        # This function now expects an absolute path.
        # The caller (e.g., API endpoint) is responsible for resolving it.
        logger.error(
            f"parse_ini_file expects an absolute path, but received: {ini_path}"
        )
        raise ValueError(f"INI path must be absolute. Received: {ini_path}")

    cached_data = ini_cache.get(ini_path)
    if cached_data:
        logger.debug(f"Returning cached INI data for: {ini_path}")
        return cached_data.copy()  # Return a copy

    if not os.path.exists(ini_path):
        logger.error(f"INI file not found: {ini_path}")
        raise FileNotFoundError(f"INI file not found: {ini_path}")

    try:
        config = configparser.ConfigParser()
        # Ensure empty lines or lines with only whitespace are ignored if not section headers or options
        config.read(ini_path, encoding="utf-8")

        if "MatchSettings" not in config:
            logger.error(f"Missing [MatchSettings] section in INI file: {ini_path}")
            raise ValueError(f"Missing [MatchSettings] section in INI file: {ini_path}")

        settings: Dict[str, Any] = {}
        expected_types = {
            "template_path": str,
            "filter_type": str,
            "match_method": str,
            "threshold": float,
            "match_range_x1": int,
            "match_range_y1": int,
            "match_range_x2": int,
            "match_range_y2": int,
            "offset_x": int,
            "offset_y": int,
            "waitforrecheck": float,  # Assuming this is a float
            "canny_t1": int,
            "canny_t2": int,
        }

        match_settings_section = config["MatchSettings"]
        for key, value_str in match_settings_section.items():
            value_str = value_str.strip()

            if key not in expected_types:
                logger.warning(
                    f"INI key '{key}' in [MatchSettings] is not in expected_types, keeping as string: '{value_str}' for {ini_path}"
                )
                settings[key] = value_str
                continue

            target_type = expected_types[key]

            if value_str == "":
                # Handle empty string based on expected type
                if target_type is str:
                    settings[key] = ""  # Empty string is valid for string type
                else:
                    # For non-string types, an empty value is an error unless explicitly allowed to be None
                    # For now, consider it an error.
                    logger.error(
                        f"Empty value for key '{key}' (expected {target_type.__name__}) in INI: {ini_path}"
                    )
                    raise ValueError(
                        f"Empty value for key '{key}' in INI file '{ini_path}' is not valid for expected type {target_type.__name__}."
                    )
                continue

            try:
                if target_type is bool:  # configparser's getboolean is robust
                    settings[key] = match_settings_section.getboolean(key)
                elif target_type is str:
                    settings[key] = value_str  # Already a string
                else:  # int, float
                    settings[key] = target_type(value_str)
            except ValueError as e:
                logger.error(
                    f"Invalid value for key '{key}' ('{value_str}') in INI '{ini_path}'. Expected {target_type.__name__}. Error: {e}"
                )
                raise ValueError(
                    f"Invalid value for key '{key}' ('{value_str}') in INI file '{ini_path}'. Expected type {target_type.__name__}."
                ) from e

        # Validate and resolve template_path
        if "template_path" not in settings or not settings["template_path"]:
            logger.error(
                f"Required 'template_path' missing or empty in [MatchSettings] of INI: {ini_path}"
            )
            raise ValueError(
                f"Required 'template_path' missing or empty in [MatchSettings] of INI file: {ini_path}"
            )

        tp_from_ini = settings["template_path"]
        if not os.path.isabs(tp_from_ini):
            # Resolve relative to the INI file's directory
            resolved_tp = os.path.abspath(
                os.path.join(os.path.dirname(ini_path), tp_from_ini)
            )
        else:
            resolved_tp = os.path.normpath(tp_from_ini)

        if not os.path.exists(resolved_tp):
            logger.error(
                f"Template file specified in INI ('{tp_from_ini}', resolved to '{resolved_tp}') not found. INI: {ini_path}"
            )
            raise FileNotFoundError(
                f"Template file '{resolved_tp}' (from INI: {ini_path}) not found."
            )

        settings["template_path"] = resolved_tp
        settings["ini_path"] = (
            ini_path  # Store the original (absolute) INI path for reference
        )

        ini_cache.put(ini_path, settings)
        return settings.copy()  # Return a copy

    except configparser.Error as e:
        logger.error(f"Error parsing INI file '{ini_path}': {e}")
        raise ValueError(f"Error parsing INI file '{ini_path}': {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected error parsing INI file '{ini_path}': {e}")
        # Re-raise as a generic ValueError or a custom INIParsingError if defined
        raise ValueError(f"Unexpected error parsing INI file '{ini_path}': {e}") from e


def clean_debug_images(debug_dir: str, max_files: int = 100) -> None:
    if not os.path.exists(debug_dir):
        return
    try:
        files = [
            os.path.join(debug_dir, f)
            for f in os.listdir(debug_dir)
            if os.path.isfile(os.path.join(debug_dir, f))
        ]
        if len(files) <= max_files:
            return
        files.sort(key=os.path.getmtime)
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
    frame: np.ndarray, debug_dir: str, prefix: str = "frame", max_files: int = 100
) -> Optional[str]:
    try:
        os.makedirs(debug_dir, exist_ok=True)
        clean_debug_images(debug_dir, max_files)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"{prefix}_{timestamp}_{ms:03d}.jpg"
        filepath = os.path.join(debug_dir, filename)

        # Use imencode and write bytes to handle potential unicode paths robustly
        retval, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )  # Added quality param
        if retval:
            with open(filepath, "wb") as f:
                f.write(buffer)
            logger.debug(f"Saved debug frame to {filepath}")
            return filepath
        else:
            logger.error(f"Failed to encode debug frame for saving: {filepath}")
            return None
    except Exception as e:
        logger.error(f"Error saving debug frame: {e}")
        return None


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is in use on a given host."""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)  # Add a short timeout
        s.connect((host, port))
        return True
    except (socket.timeout, ConnectionRefusedError):
        return False
    except Exception as e:
        logger.debug(f"Error checking port {host}:{port}: {e}")
        return False  # Assume not in use or unable to check
    finally:
        if s:
            s.close()
