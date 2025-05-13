# -*- coding: utf-8 -*-
"""
Custom data types and validation utilities for image search server.
"""
import logging
import os
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np  # Import numpy for array type hinting

logger = logging.getLogger(__name__)


class FilterType(str, Enum):
    """Image filter types with validation"""

    NONE = "none"
    CANNY = "canny"

    @classmethod
    def validate(cls, value: Any) -> "FilterType":
        """Validate and convert input to FilterType"""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except KeyError:
                pass
            try:
                return cls(value.lower())
            except ValueError:
                pass
        raise ValueError(
            f"Invalid filter type: {value}. Must be one of: {', '.join(t.value for t in cls)}"
        )


class MatchMethod(str, Enum):
    """Template matching algorithms with validation"""

    CCoeffNormed = "ccoeff_normed"
    SqDiffNormed = "sqdiff_normed"
    CCorrNormed = "ccorr_normed"
    SqDiff = "sqdiff"
    CCoeff = "ccoeff"
    CCorr = "ccorr"

    @classmethod
    def validate(cls, value: Any) -> "MatchMethod":
        """Validate and convert input to MatchMethod"""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.split(".")[-1].strip().lower()]
            except (KeyError, IndexError):
                pass
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise ValueError(
            f"Invalid match method: {value}. Must be one of: {', '.join(t.value for t in cls)}"
        )


class TemplateMatchResult:
    """Immutable container for template matching results"""

    def __init__(
        self,
        success: bool,
        template_path: str,
        score: float,
        threshold: float,
        center_coords: Tuple[int, int],
        top_left_coords: Tuple[int, int],
        size: Tuple[int, int],
        offset_coords: Tuple[int, int],
        verify_result: Optional[Tuple[bool, float]] = None,
        processing_details: Optional[Dict[str, Any]] = None,
        frame_info: Optional[Dict[str, Any]] = None,
    ):
        self._validate_inputs(
            success,
            score,
            threshold,
            center_coords,
            top_left_coords,
            size,
            offset_coords,
        )

        self.success = success
        self.template_path = str(template_path)
        self.score = float(score)
        self.threshold = float(threshold)
        self.center_coords = tuple(center_coords)
        self.top_left_coords = tuple(top_left_coords)
        self.size = tuple(size)
        self.offset_coords = tuple(offset_coords)
        self.verify_result = verify_result
        self.processing_details = processing_details or {}
        self.frame_info = frame_info or {}

    def _validate_inputs(
        self,
        success: bool,
        score: float,
        threshold: float,
        center_coords: Tuple[int, ...],
        top_left_coords: Tuple[int, ...],
        size: Tuple[int, ...],
        offset_coords: Tuple[int, ...],
    ) -> None:
        """Validate input parameters"""
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")

        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("score must be a float between 0 and 1")

        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a float between 0 and 1")

        if len(center_coords) != 2 or not all(
            isinstance(x, int) for x in center_coords
        ):
            raise ValueError("center_coords must be a tuple of two integers")

        if len(top_left_coords) != 2 or not all(
            isinstance(x, int) for x in top_left_coords
        ):
            raise ValueError("top_left_coords must be a tuple of two integers")

        if len(size) != 2 or not all(isinstance(x, int) and x >= 0 for x in size):
            raise ValueError("size must be a tuple of two non-negative integers")

        if len(offset_coords) != 2 or not all(
            isinstance(x, int) for x in offset_coords
        ):
            raise ValueError("offset_coords must be a tuple of two integers")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        result = {
            "success": self.success,
            "template_path": self.template_path,
            "score": self.score,
            "threshold": self.threshold,
            "center_coords": list(self.center_coords),
            "top_left_coords": list(self.top_left_coords),
            "size": list(self.size),
            "offset_coords": list(self.offset_coords),
            "processing_details": self.processing_details,
            "frame_info": self.frame_info,
        }

        if self.verify_result:
            result["verify_confirmed"] = self.verify_result[0]
            result["verify_score"] = self.verify_result[1]

        return result


class MJPEGStreamStatus:
    """Immutable container for MJPEG stream status information"""

    def __init__(
        self,
        active: bool,
        ready: bool,
        frame_age: float,
        resolution: Tuple[int, int],
        status_message: str,
        error_count: int = 0,
    ):
        self._validate_inputs(frame_age, resolution)

        self.active = bool(active)
        self.ready = bool(ready)
        self.frame_age = float(frame_age)
        self.resolution = tuple(resolution)
        self.status_message = str(status_message)
        self.error_count = int(error_count)

    def _validate_inputs(self, frame_age: float, resolution: Tuple[int, int]) -> None:
        """Validate input parameters"""
        if not isinstance(frame_age, (int, float)) or frame_age < 0:
            raise ValueError("frame_age must be a non-negative number")

        if len(resolution) != 2 or not all(
            isinstance(x, int) and x > 0 for x in resolution
        ):
            raise ValueError("resolution must be a tuple of two positive integers")

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary format"""
        return {
            "active": self.active,
            "ready": self.ready,
            "frame_age_seconds": self.frame_age,
            "frame_resolution": f"{self.resolution[0]}x{self.resolution[1]}",
            "status": self.status_message,
            "error_count": self.error_count,
        }


# New: NamedTuple for Template Cache Entry
class TemplateCacheEntry(NamedTuple):
    """Data structure for an entry in the TemplateCache."""

    processed_template: (
        np.ndarray
    )  # The processed template image (e.g., grayscale, filtered)
    width: int  # Width of the processed template
    height: int  # Height of the processed template
    processing_details: str  # String describing the processing applied (e.g., "Filter: none", "Filter: canny (Params: ...)").


# 自定义异常类
class ImageSearchError(Exception):
    """图像搜索基础错误类"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.error_type = self.__class__.__name__


class ValidationError(ImageSearchError):
    """参数验证错误"""


class ResourceNotFoundError(ImageSearchError):
    """资源未找到错误"""


class FilterProcessingError(ImageSearchError):
    """图像过滤处理错误"""


class TemplateLoadError(ImageSearchError):
    """模板加载错误"""


class TemplateMatchError(ImageSearchError):
    """模板匹配错误"""


class InvalidParameterError(ImageSearchError):
    """无效参数错误"""


class MJPEGStreamError(ImageSearchError):
    """MJPEG 流错误"""


class CacheError(ImageSearchError):
    """缓存操作错误"""


# 验证类
class ParameterValidator:
    """参数验证器"""

    @staticmethod
    def validate_threshold(value: float) -> float:
        """验证阈值参数"""
        if not isinstance(value, (int, float)):
            raise ValidationError("阈值必须是数字类型")
        if not 0 <= value <= 1:
            raise ValidationError("阈值必须在 0 到 1 之间")
        return float(value)

    @staticmethod
    def validate_coordinates(
        x: Optional[int], y: Optional[int], width: int, height: int
    ) -> Tuple[int, int]:
        """验证坐标值"""
        if x is None or y is None:
            return (0, 0)

        if not isinstance(x, int) or not isinstance(y, int):
            raise ValidationError("坐标值必须是整数")

        if x < 0 or y < 0:
            raise ValidationError("坐标值不能为负数")

        if x >= width or y >= height:
            raise ValidationError(f"坐标值超出范围 (width: {width}, height: {height})")

        return (x, y)

    @staticmethod
    def validate_search_region(
        x1: Optional[int],
        y1: Optional[int],
        x2: Optional[int],
        y2: Optional[int],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, int]:
        """验证搜索区域参数"""
        if all(v is None for v in (x1, y1, x2, y2)):
            return {"x1": 0, "y1": 0, "x2": frame_width, "y2": frame_height}

        if any(v is None for v in (x1, y1, x2, y2)):
            raise ValidationError("搜索区域参数必须全部提供或全部省略")

        if not all(isinstance(v, int) for v in (x1, y1, x2, y2)):
            raise ValidationError("搜索区域参数必须是整数")

        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise ValidationError("搜索区域参数不能为负数")

        if x1 >= x2 or y1 >= y2:
            raise ValidationError("搜索区域参数必须满足: x1 < x2 且 y1 < y2")

        if x2 > frame_width or y2 > frame_height:
            raise ValidationError(
                f"搜索区域超出图像范围 (width: {frame_width}, height: {frame_height})"
            )

        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @staticmethod
    def validate_canny_params(t1: Optional[int], t2: Optional[int]) -> Tuple[int, int]:
        """验证 Canny 边缘检测参数"""
        if t1 is None and t2 is None:
            return (100, 200)  # 默认值

        if t1 is None or t2 is None:
            raise ValidationError("Canny 参数必须同时提供或同时省略")

        if not isinstance(t1, int) or not isinstance(t2, int):
            raise ValidationError("Canny 参数必须是整数")

        if t1 < 0 or t2 < 0:
            raise ValidationError("Canny 参数不能为负数")

        if t1 >= t2:
            raise ValidationError("必须满足 t1 < t2")

        if t2 > 255:
            raise ValidationError("t2 不能超过 255")

        return (t1, t2)

    @staticmethod
    def validate_image_format(image_path: str) -> str:
        """验证图像格式是否支持"""
        valid_formats = [".png", ".jpg", ".jpeg", ".bmp"]
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in valid_formats:
            raise ValidationError(
                f"不支持的图像格式: {ext}。支持的格式: {', '.join(valid_formats)}"
            )
        return image_path

    @staticmethod
    def validate_mjpeg_url(url: str) -> str:
        """验证 MJPEG URL 格式"""
        if not url.startswith(("http://", "https://", "rtsp://")):
            raise ValidationError("MJPEG URL 必须以 http://, https:// 或 rtsp:// 开头")
        return url
