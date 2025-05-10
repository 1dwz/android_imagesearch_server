# -*- coding: utf-8 -*-
"""
Classes for managing shared state: FrameDataCache, TemplateCache, MJPEGStreamManager.
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    CACHE_KEY_HASH_LENGTH,
    DEFAULT_CANNY_T1,
    DEFAULT_CANNY_T2,
    DEFAULT_TEMPLATE_CACHE_SIZE,
)
from .datatypes import FilterType

logger = logging.getLogger(__name__)


# --- FrameDataCache ---
class FrameDataCache:
    """
    Thread-safe cache for the latest frame data.
    """

    def __init__(self):
        # Frame data storage
        self._frame_data: Dict[str, Any] = {
            "gray_frame": None,
            "frame_copy": None,  # Color frame copy for debugging/annotation
            "shape": None,
            "timestamp": 0.0,
            "status": "Not Started",
            "timestamp_str": "N/A",  # Human-readable timestamp string
            "last_accessed": time.monotonic(),  # Timestamp of last access
        }
        self._lock = threading.Lock()
        logger.debug("FrameDataCache initialized.")

    def update_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        status: str = "Running",
    ) -> None:
        """
        Update the cached frame data.
        """
        logger.debug(
            f"Attempting to update frame data. Timestamp: {timestamp}, Status: {status}"
        )

        # Perform CPU-intensive operations (cvtColor, copy) outside the lock
        gray_frame_processed = None
        frame_copy_processed = None
        frame_shape_processed = None

        if frame is not None:
            try:
                gray_frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except cv2.error as e:
                logger.warning(f"Error converting frame to grayscale: {e}")
                # Decide how to handle conversion errors. For now, gray_frame_processed will remain None.

            try:
                frame_copy_processed = frame.copy()
            except Exception as e:
                logger.warning(f"Error copying frame: {e}")
                # Decide how to handle copy errors. For now, frame_copy_processed will remain None.

            frame_shape_processed = frame.shape

        with self._lock:
            # Create copies to avoid external modifications affecting the cached data
            # Assign the already processed data within the lock
            self._frame_data["gray_frame"] = gray_frame_processed
            self._frame_data["frame_copy"] = frame_copy_processed
            self._frame_data["shape"] = frame_shape_processed
            self._frame_data["timestamp"] = timestamp
            self._frame_data["status"] = status
            self._frame_data["timestamp_str"] = datetime.fromtimestamp(
                timestamp
            ).strftime("%H:%M:%S.%f")[:-3]
            self._frame_data["last_accessed"] = time.monotonic()
            logger.debug(
                f"Frame data updated in cache. New timestamp: {self._frame_data['timestamp_str']}, Shape: {self._frame_data['shape']}, Status: {self._frame_data['status']}"
            )

    def update_status(self, status: str) -> None:
        """
        Update only the status message.
        """
        logger.debug(f"Attempting to update frame cache status to: {status}")
        with self._lock:
            self._frame_data["status"] = status
            self._frame_data["last_accessed"] = time.monotonic()
            logger.debug(f"Frame cache status updated to: {status}")

    def get_data(self) -> Dict[str, Any]:
        """
        Get a thread-safe copy of the cached frame data.
        """
        logger.debug("Attempting to get frame data from cache.")
        with self._lock:
            # Return a copy to prevent external modification of internal state
            data = self._frame_data.copy()
            data["gray_frame"] = (
                self._frame_data["gray_frame"].copy()
                if self._frame_data["gray_frame"] is not None
                else None
            )
            data["frame_copy"] = (
                self._frame_data["frame_copy"].copy()
                if self._frame_data["frame_copy"] is not None
                else None
            )
            data["last_accessed"] = time.monotonic()
            logger.debug(
                f"Retrieved frame data from cache. Status: {data.get('status')}, Timestamp: {data.get('timestamp_str')}"
            )
            return data

    def get_frame_data(self) -> Tuple[Dict[str, Any], float, str]:
        """
        Get the latest frame data including status and timestamp.
        This is a simplified version for status checks.
        """
        logger.debug("Attempting to get simplified frame data for status check.")
        with self._lock:
            # Return copies or simple types for safety
            frame_data_copy = self._frame_data.copy()
            timestamp = self._frame_data["timestamp"]
            status = self._frame_data["status"]
            # Optionally include timestamp_str for display
            timestamp_str = self._frame_data["timestamp_str"]
            last_accessed = self._frame_data["last_accessed"]
            logger.debug(
                f"Retrieved simplified frame data. Status: {status}, Timestamp: {timestamp_str}, Last accessed: {last_accessed}"
            )
            return frame_data_copy, timestamp, status

    def reset_idle_timer(self) -> None:
        """
        Reset the last accessed timestamp to indicate activity.
        """
        logger.debug("Resetting frame cache idle timer.")
        with self._lock:
            self._frame_data["last_accessed"] = time.monotonic()
            logger.debug(
                f"Frame cache idle timer reset. last_accessed={self._frame_data['last_accessed']}"
            )

    def is_idle(self, idle_timeout_seconds: float) -> bool:
        """
        Check if the cache has been idle for longer than the specified timeout.
        """
        logger.debug(
            f"Checking if frame cache is idle. Timeout: {idle_timeout_seconds:.2f}s"
        )
        if idle_timeout_seconds == 9999999.0 or idle_timeout_seconds > 9999999.0:
            logger.debug("Idle timeout is very large. Cache is never considered idle.")
            return False  # Never idle if timeout is very large

        with self._lock:
            idle_time = time.monotonic() - self._frame_data["last_accessed"]
            is_idle = idle_time > idle_timeout_seconds
            logger.debug(
                f"Frame cache idle check: Idle for {idle_time:.2f}s, Timeout: {idle_timeout_seconds:.2f}s, Is idle: {is_idle}"
            )
            return is_idle


# --- TemplateCache ---
class TemplateCache:
    """Thread-safe LRU cache for processed template images."""

    def __init__(self, max_size: int = DEFAULT_TEMPLATE_CACHE_SIZE):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        logger.debug(f"TemplateCache initialized with max size: {self._max_size}")

    def _generate_cache_key(
        self, tpl_path: str, filter_type: FilterType, filter_params: Dict[str, Any]
    ) -> str:
        """Generate a unique key for the template based on path and processing params."""
        # Start with the template path
        key_elements = [str(tpl_path)]

        # Add filter type
        key_elements.append(filter_type.value)

        # Add filter parameters if applicable
        if filter_type == FilterType.CANNY:
            # Extract and add parameters specific to Canny filter
            canny_t1 = filter_params.get("canny_t1", DEFAULT_CANNY_T1)
            canny_t2 = filter_params.get("canny_t2", DEFAULT_CANNY_T2)
            key_elements.append(f"t1={canny_t1}")
            key_elements.append(f"t2={canny_t2}")

        # Join all elements with a separator
        key_str = "|".join(key_elements)

        # Create a hash of the key string for shorter keys
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:CACHE_KEY_HASH_LENGTH]

        # Final key format: tpl_filename|filter_type|hash
        final_key = f"{Path(tpl_path).stem}|{filter_type.value}|{key_hash}"
        logger.debug(f"Generated cache key: {final_key} for template {tpl_path}")
        return final_key

    def store_template(
        self,
        tpl_path: str,
        filter_type: FilterType,
        filter_params: Dict[str, Any],
        template_data: Tuple[np.ndarray, int, int],
    ) -> None:
        """Store a processed template in the cache with the given parameters."""
        key = self._generate_cache_key(tpl_path, filter_type, filter_params)
        with self._lock:
            # Remove if exists to update position in OrderedDict (LRU)
            if key in self._cache:
                self._cache.pop(key)

            # Add to cache
            self._cache[key] = template_data
            logger.debug(f"Template cached: {key}")

            # Trim cache if necessary
            if len(self._cache) > self._max_size:
                # OrderedDict remembers insertion order, so first item is oldest
                oldest_key, _ = self._cache.popitem(last=False)
                logger.debug(
                    f"Cache limit reached, removed oldest template: {oldest_key}"
                )

    def get_template(
        self, tpl_path: str, filter_type: FilterType, filter_params: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """Retrieve a processed template from the cache if it exists."""
        key = self._generate_cache_key(tpl_path, filter_type, filter_params)
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                template_data = self._cache.pop(key)
                self._cache[key] = template_data
                logger.debug(f"Template cache hit: {key}")
                return template_data
            logger.debug(f"Template cache miss: {key}")
            return None

    # 添加更简单的API以匹配新的模板处理逻辑
    def get(self, key: str) -> Optional[Tuple[np.ndarray, int, int, str]]:
        """
        获取缓存中的模板数据。

        Args:
            key: 缓存键

        Returns:
            如果命中，返回缓存的模板数据，否则返回None
        """
        with self._lock:
            if key in self._cache:
                # 更新LRU顺序
                template_data = self._cache.pop(key)
                self._cache[key] = template_data
                logger.debug(f"Template cache hit: {key}")
                return template_data
            logger.debug(f"Template cache miss: {key}")
            return None

    def set(self, key: str, template_data: Tuple[np.ndarray, int, int, str]) -> None:
        """
        将处理后的模板存储到缓存中。

        Args:
            key: 缓存键
            template_data: 模板数据，包含(处理后图像, 原始高度, 原始宽度, 处理描述)
        """
        with self._lock:
            # 如果已存在，先移除以更新LRU顺序
            if key in self._cache:
                self._cache.pop(key)

            # 添加到缓存
            self._cache[key] = template_data
            logger.debug(f"Template cached: {key}")

            # 如有必要，移除最旧的条目
            if len(self._cache) > self._max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                logger.debug(
                    f"Cache limit reached, removed oldest template: {oldest_key}"
                )

    def clear(self) -> None:
        """Clear all entries from the cache."""
        logger.debug("Clearing template cache")
        with self._lock:
            self._cache.clear()
            logger.debug("Template cache cleared")

    def get_size(self) -> int:
        """Get the current number of items in the cache."""
        with self._lock:
            size = len(self._cache)
            logger.debug(f"Template cache current size: {size}")
            return size


# --- MJPEGStreamManager ---
# The class is now imported from modules.mjpeg_reader
