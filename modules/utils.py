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

import cv2
import numpy as np

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
    解析 INI 文件并提取设置

    Args:
        ini_path: INI 文件的路径

    Returns:
        包含 [MatchSettings] 部分设置的字典

    Raises:
        FileNotFoundError: 如果 INI 文件不存在
        ValueError: 如果 INI 文件缺少必需的部分或数据无效
    """
    # 首先检查缓存
    cached_data = ini_cache.get(ini_path)
    if cached_data:
        return cached_data

    # 去除路径中可能的空白和换行符
    ini_path = ini_path.strip()

    # 检查文件是否存在
    if not os.path.exists(ini_path):
        logger.error(f"未找到 INI 文件: {ini_path}")
        raise FileNotFoundError(f"未找到 INI 文件: {ini_path}")

    try:
        # 创建解析器并读取文件
        config = configparser.ConfigParser()
        config.read(ini_path, encoding="utf-8")

        # 检查是否存在 [MatchSettings] 部分
        if "MatchSettings" not in config:
            logger.error(f"在 {ini_path} 中缺少 [MatchSettings] 部分")
            raise ValueError(f"在 {ini_path} 中缺少 [MatchSettings] 部分")

        # 从 [MatchSettings] 部分提取设置并进行类型转换
        settings = {}
        # 定义预期的类型映射
        expected_types = {
            "threshold": float,
            "match_range_x1": int,
            "match_range_y1": int,
            "match_range_x2": int,
            "match_range_y2": int,
            "offset_x": int,
            "offset_y": int,
            "waitforrecheck": float,
            # 注意：server.ini 中的类型转换应该在读取 server.ini 的地方处理，
            # 如果 parse_ini_file 仅用于 MatchSettings，则此处只包含 MatchSettings 的键。
            # 如果 parse_ini_file 被设计为通用，则需要更复杂的类型映射逻辑。
            # 假设它主要用于 MatchSettings:
            "canny_t1": int,
            "canny_t2": int,
        }

        for key, value_str in config["MatchSettings"].items():
            value_str = value_str.strip()
            if value_str == "":
                settings[key] = None
            elif key in expected_types:
                try:
                    # 尝试将值转换为期望的类型
                    settings[key] = expected_types[key](value_str)
                except ValueError as e:
                    logger.warning(
                        f"无法将 INI 值 '{value_str}' (键: {key}) 转换为类型 {expected_types[key]}. 错误: {e}. 保留为字符串."
                    )
                    settings[key] = value_str  # 转换失败时保留为字符串
            else:
                # 对于没有定义预期类型的键，保留为字符串
                settings[key] = value_str

        # 检查是否已在 INI 文件中指定了 template_path
        if "template_path" not in settings or not settings["template_path"]:
            # 如果未指定，则派生模板路径（将 .ini 替换为 .jpg）
            ini_basename = os.path.basename(ini_path)
            template_name = os.path.splitext(ini_basename)[0] + ".jpg"
            template_dir = os.path.dirname(ini_path)
            template_path = os.path.join(template_dir, template_name)
            settings["template_path"] = template_path
            logger.debug(f"使用派生的模板路径: {template_path}")
        else:
            # 如果指定了相对路径，将其转换为相对于 INI 文件的绝对路径
            template_path = settings["template_path"]
            if not os.path.isabs(template_path):
                template_path = os.path.join(os.path.dirname(ini_path), template_path)
                settings["template_path"] = template_path
            logger.debug(f"使用 INI 文件中指定的模板路径: {template_path}")

        settings["ini_path"] = ini_path

        # 缓存解析的数据
        ini_cache.put(ini_path, settings)

        return settings

    except configparser.Error as e:
        logger.error(f"解析 INI 文件 {ini_path} 时出错: {e}")
        raise ValueError(f"解析 INI 文件时出错: {e}")
    except Exception as e:
        logger.error(f"解析 INI 文件 {ini_path} 时发生意外错误: {e}")
        raise


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
    frame: np.ndarray,
    debug_dir: str,
    prefix: str = "frame",
    max_files: int = 100,
) -> Optional[str]:
    """
    保存调试帧图像到指定目录。

    Args:
        frame: 要保存的帧图像
        debug_dir: 保存目录
        prefix: 文件名前缀
        max_files: 目录中保留的最大文件数

    Returns:
        保存的文件路径，如果保存失败则返回 None
    """
    try:
        # 确保目录存在
        os.makedirs(debug_dir, exist_ok=True)

        # 清理旧文件
        clean_debug_images(debug_dir, max_files)

        # 生成带时间戳的文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"{prefix}_{timestamp}_{ms:03d}.jpg"
        filepath = os.path.join(debug_dir, filename)

        # 使用 imencode 和 tofile 来处理 Unicode 路径
        retval, buffer = cv2.imencode(".jpg", frame)
        if retval:
            with open(filepath, "wb") as f:
                buffer.tofile(f)
            logger.debug(f"已保存调试帧到 {filepath}")
            return filepath
        else:
            logger.error(f"编码调试帧失败: {filepath}")
            return None

    except Exception as e:
        logger.error(f"保存调试帧时出错: {e}")
        return None
