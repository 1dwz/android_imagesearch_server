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
from .datatypes import CacheError, FilterType, TemplateCacheEntry

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
        """
        生成基于模板路径和处理参数的唯一缓存键。

        Args:
            tpl_path: 模板路径
            filter_type: 过滤器类型
            filter_params: 过滤器参数

        Returns:
            缓存键字符串

        Raises:
            CacheError: 当生成缓存键失败时
        """
        try:
            # 开始构建键元素
            key_elements = [str(tpl_path), filter_type.value]

            # 添加过滤器参数
            if filter_type == FilterType.CANNY:
                t1 = filter_params.get("canny_t1", DEFAULT_CANNY_T1)
                t2 = filter_params.get("canny_t2", DEFAULT_CANNY_T2)
                key_elements.extend([str(t1), str(t2)])

            # 生成键字符串
            key_str = "|".join(key_elements)

            # 创建哈希
            key_hash = hashlib.sha256(key_str.encode()).hexdigest()[
                :CACHE_KEY_HASH_LENGTH
            ]

            return f"{key_hash}_{Path(tpl_path).name}"

        except Exception as e:
            raise CacheError(f"生成缓存键时出错: {e}")

    def store_template(
        self,
        tpl_path: str,
        filter_type: FilterType,
        filter_params: Dict[str, Any],
        template_data: Tuple[np.ndarray, int, int],
    ) -> None:
        """
        将处理过的模板存储到缓存中。

        Args:
            tpl_path: 模板路径
            filter_type: 过滤器类型
            filter_params: 过滤器参数
            template_data: 模板数据元组 (processed_template, width, height)

        Raises:
            CacheError: 当存储模板失败时
        """
        try:
            key = self._generate_cache_key(tpl_path, filter_type, filter_params)
            with self._lock:
                # 检查缓存大小并在需要时清除旧条目
                if len(self._cache) >= self._max_size:
                    # 移除最旧的条目
                    self._cache.popitem(last=False)
                    logger.debug("已从缓存中移除最旧的模板")

                # 存储新模板
                self._cache[key] = TemplateCacheEntry(
                    processed_template=template_data[0],
                    width=template_data[1],
                    height=template_data[2],
                    processing_details=f"Filter: {filter_type.value}, Params: {filter_params}",
                )
                logger.debug(f"已将模板存储到缓存中，键: {key}")

        except Exception as e:
            raise CacheError(f"存储模板到缓存时出错: {e}")

    def get_template(
        self, tpl_path: str, filter_type: FilterType, filter_params: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """
        从缓存获取处理过的模板。

        Args:
            tpl_path: 模板路径
            filter_type: 过滤器类型
            filter_params: 过滤器参数

        Returns:
            模板数据元组 (processed_template, width, height) 或 None

        Raises:
            CacheError: 当获取模板失败时
        """
        try:
            key = self._generate_cache_key(tpl_path, filter_type, filter_params)
            with self._lock:
                if key in self._cache:
                    # 移动到最近使用的位置
                    entry = self._cache.pop(key)
                    self._cache[key] = entry
                    logger.debug(f"从缓存中获取到模板，键: {key}")
                    return (entry.processed_template, entry.width, entry.height)
                logger.debug(f"缓存中未找到模板，键: {key}")
                return None

        except Exception as e:
            raise CacheError(f"从缓存获取模板时出错: {e}")

    def clear(self) -> None:
        """
        清除缓存中的所有条目。

        Raises:
            CacheError: 当清除缓存失败时
        """
        try:
            with self._lock:
                self._cache.clear()
                logger.debug("已清除模板缓存")
        except Exception as e:
            raise CacheError(f"清除缓存时出错: {e}")

    def get_size(self) -> int:
        """
        获取当前缓存大小。

        Returns:
            缓存中的条目数量

        Raises:
            CacheError: 当获取缓存大小失败时
        """
        try:
            with self._lock:
                size = len(self._cache)
                logger.debug(f"当前缓存大小: {size}")
                return size
        except Exception as e:
            raise CacheError(f"获取缓存大小时出错: {e}")


# --- MJPEGStreamManager ---
# The class is now imported from modules.mjpeg_reader
