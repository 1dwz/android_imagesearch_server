# -*- coding: utf-8 -*-
"""
Classes for managing shared state: FrameDataCache, TemplateCache.
MJPEGStreamManager is now directly modules.mjpeg.MJPEGStreamReader.
"""
import hashlib
import logging
import os  # <--- BUG FIX: Added import os
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .constants import INFINITE_TIMEOUT  # Updated import
from .constants import (
    CACHE_KEY_HASH_LENGTH,
    DEFAULT_CANNY_T1,
    DEFAULT_CANNY_T2,
    DEFAULT_TEMPLATE_CACHE_SIZE,
)
from .datatypes import CacheError, FilterType, TemplateCacheEntry

logger = logging.getLogger(__name__)


class FrameDataCache:
    """Thread-safe cache for the latest frame data from an MJPEG stream."""

    def __init__(self):
        self._frame_data: Dict[str, Any] = {
            "gray_frame": None,
            "frame_copy": None,
            "shape": None,  # Tuple (height, width, channels) or (height, width) for grayscale
            "timestamp": 0.0,  # Epoch timestamp of the frame
            "status": "Not Started",  # Stream status related to this frame
            "timestamp_str": "N/A",  # Human-readable
            "last_accessed": time.monotonic(),  # For idle checking
        }
        self._lock = threading.Lock()
        logger.debug("FrameDataCache initialized.")

    def update_frame(
        self, frame: Optional[np.ndarray], timestamp: float, status: str = "Running"
    ) -> None:
        """
        Updates the cached frame data. If frame is None, updates status and timestamp only.
        """
        # logger.debug(f"Attempting to update frame data. Timestamp: {timestamp}, Status: {status}")
        gray_frame_processed = None
        frame_copy_processed = None
        frame_shape_processed = None

        if frame is not None:
            try:
                # Ensure frame is BGR before converting to GRAY for consistency
                if len(frame.shape) == 2:  # Already grayscale
                    gray_frame_processed = frame.copy()  # Still copy
                elif frame.shape[2] == 3:  # BGR
                    gray_frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                elif frame.shape[2] == 4:  # BGRA
                    gray_frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                else:
                    raise ValueError(
                        f"Unsupported frame channel count: {frame.shape[2]}"
                    )

                frame_copy_processed = (
                    frame.copy()
                )  # Store a copy of the original (color) frame
                frame_shape_processed = frame.shape
            except cv2.error as e:
                logger.warning(f"Error processing frame for cache (cv2 error): {e}")
                # Keep processed versions as None, status will reflect this
            except ValueError as e:
                logger.warning(f"Error processing frame for cache (value error): {e}")

        with self._lock:
            self._frame_data["gray_frame"] = gray_frame_processed
            self._frame_data["frame_copy"] = frame_copy_processed
            self._frame_data["shape"] = frame_shape_processed
            self._frame_data["timestamp"] = timestamp
            self._frame_data["status"] = (
                status
                if frame is not None and gray_frame_processed is not None
                else "Frame Error"
            )
            self._frame_data["timestamp_str"] = datetime.fromtimestamp(
                timestamp
            ).strftime("%H:%M:%S.%f")[:-3]
            self._frame_data["last_accessed"] = time.monotonic()
            # logger.debug(
            #     f"Frame data updated. Timestamp: {self._frame_data['timestamp_str']}, "
            #     f"Shape: {self._frame_data['shape']}, Status: {self._frame_data['status']}"
            # )

    def update_status(self, status: str, timestamp: Optional[float] = None) -> None:
        """Updates only the status message and optionally the timestamp."""
        with self._lock:
            self._frame_data["status"] = status
            if timestamp is not None:
                self._frame_data["timestamp"] = timestamp
                self._frame_data["timestamp_str"] = datetime.fromtimestamp(
                    timestamp
                ).strftime("%H:%M:%S.%f")[:-3]
            self._frame_data["last_accessed"] = time.monotonic()
            logger.debug(f"Frame cache status updated to: {status}")

    def get_data(self) -> Dict[str, Any]:
        """
        Get a thread-safe copy of all cached frame data.
        Consumers should check 'gray_frame' and 'frame_copy' for None.
        """
        with self._lock:
            # Shallow copy of the dict; numpy arrays inside are already copies from update_frame
            data_to_return = self._frame_data.copy()
            self._frame_data["last_accessed"] = time.monotonic()
            # logger.debug(f"Retrieved frame data. Status: {data_to_return.get('status')}")
            return data_to_return

    def get_current_frame(
        self,
    ) -> Tuple[Optional[np.ndarray], float, Optional[Tuple[int, int, Optional[int]]]]:
        """
        Gets a copy of the current color frame, its timestamp, and shape.
        Returns (None, 0.0, None) if no valid frame.
        """
        with self._lock:
            self._frame_data["last_accessed"] = time.monotonic()
            if self._frame_data["frame_copy"] is not None:
                return (
                    self._frame_data["frame_copy"].copy(),
                    self._frame_data["timestamp"],
                    self._frame_data["shape"],
                )
            return None, 0.0, None

    def get_current_gray_frame(
        self,
    ) -> Tuple[Optional[np.ndarray], float, Optional[Tuple[int, int]]]:
        """
        Gets a copy of the current grayscale frame, its timestamp, and shape.
        Returns (None, 0.0, None) if no valid frame.
        """
        with self._lock:
            self._frame_data["last_accessed"] = time.monotonic()
            if self._frame_data["gray_frame"] is not None:
                shape_2d = (
                    (self._frame_data["shape"][0], self._frame_data["shape"][1])
                    if self._frame_data["shape"] and len(self._frame_data["shape"]) >= 2
                    else None
                )
                return (
                    self._frame_data["gray_frame"].copy(),
                    self._frame_data["timestamp"],
                    shape_2d,
                )
            return None, 0.0, None

    def get_status_info(self) -> Dict[str, Any]:
        """Gets status, timestamp, and shape without returning bulky frame data."""
        with self._lock:
            self._frame_data["last_accessed"] = time.monotonic()
            return {
                "status": self._frame_data["status"],
                "timestamp": self._frame_data["timestamp"],
                "timestamp_str": self._frame_data["timestamp_str"],
                "shape": self._frame_data["shape"],
                "last_accessed": self._frame_data["last_accessed"],
            }

    def reset_idle_timer(self) -> None:
        with self._lock:
            self._frame_data["last_accessed"] = time.monotonic()
            logger.debug("Frame cache idle timer reset.")

    def is_idle(self, idle_timeout_seconds: float) -> bool:
        if idle_timeout_seconds == INFINITE_TIMEOUT:
            return False
        with self._lock:
            idle_time = time.monotonic() - self._frame_data["last_accessed"]
            return idle_time > idle_timeout_seconds


class TemplateCache:
    """Thread-safe LRU cache for processed template images."""

    def __init__(self, max_size: int = DEFAULT_TEMPLATE_CACHE_SIZE):
        self._cache = OrderedDict()  # For LRU behavior
        self._max_size = max(1, max_size)  # Ensure max_size is at least 1
        self._lock = threading.Lock()
        logger.debug(f"TemplateCache initialized with max size: {self._max_size}")

    def _generate_cache_key(
        self, tpl_path: str, filter_type: FilterType, filter_params: Dict[str, Any]
    ) -> str:
        try:
            # Normalize tpl_path for consistent key generation
            norm_tpl_path = os.path.normcase(os.path.abspath(tpl_path))

            key_elements = [norm_tpl_path, filter_type.value]
            if filter_type == FilterType.CANNY:
                # Ensure consistent order and default values for Canny params in key
                t1 = filter_params.get("canny_t1", DEFAULT_CANNY_T1)
                t2 = filter_params.get("canny_t2", DEFAULT_CANNY_T2)
                key_elements.extend([str(t1), str(t2)])

            # Consider file modification time to invalidate cache if template file changes
            # This adds I/O to key generation, use with caution or make it configurable
            # try:
            #     mtime = os.path.getmtime(norm_tpl_path)
            #     key_elements.append(str(mtime))
            # except OSError: # File might not exist yet if called early
            #     pass

            key_str = "|".join(key_elements)
            # Use a shorter, but still effective hash if CACHE_KEY_HASH_LENGTH is small
            hasher = hashlib.sha256()
            hasher.update(key_str.encode("utf-8"))
            key_hash = hasher.hexdigest()[:CACHE_KEY_HASH_LENGTH]

            # Suffix with filename for easier debugging of cache keys, if desired
            # return f"{key_hash}_{Path(tpl_path).name}"
            return key_hash

        except Exception as e:
            # Log and re-raise as CacheError
            logger.error(f"Error generating template cache key for {tpl_path}: {e}")
            raise CacheError(f"Error generating template cache key: {e}")

    def store_template(
        self,
        tpl_path: str,
        filter_type: FilterType,
        filter_params: Dict[str, Any],  # Canny t1, t2 etc.
        template_entry: TemplateCacheEntry,
    ) -> None:
        if not isinstance(template_entry, TemplateCacheEntry):
            raise CacheError("Invalid template_entry type for storing in cache.")

        key = self._generate_cache_key(tpl_path, filter_type, filter_params)
        with self._lock:
            if (
                key in self._cache
            ):  # If key exists, remove to update (and move to end for LRU)
                self._cache.pop(key)
            elif len(self._cache) >= self._max_size:
                # Evict the least recently used item (first item in OrderedDict)
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"TemplateCache full. Evicted: {evicted_key}")

            self._cache[key] = template_entry
            logger.debug(f"Stored template in cache. Key: {key}, Path: {tpl_path}")

    def get_template(
        self, tpl_path: str, filter_type: FilterType, filter_params: Dict[str, Any]
    ) -> Optional[TemplateCacheEntry]:
        key = self._generate_cache_key(tpl_path, filter_type, filter_params)
        with self._lock:
            if key in self._cache:
                # Move accessed item to the end to mark as most recently used for LRU
                entry = self._cache.pop(key)
                self._cache[key] = entry
                logger.debug(
                    f"Retrieved template from cache. Key: {key}, Path: {tpl_path}"
                )
                return entry
            logger.debug(f"Template cache miss. Key: {key}, Path: {tpl_path}")
            return None

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            logger.info("TemplateCache cleared.")

    def get_size(self) -> int:
        with self._lock:
            return len(self._cache)
