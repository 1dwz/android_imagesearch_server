# -*- coding: utf-8 -*-
"""
Custom data types and validation utilities for image search server.
"""
import logging
import os
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Tuple  # Added List

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
            # Try matching by enum member name (case-insensitive) e.g. "CANNY" or "canny"
            for member_name, member_obj in cls.__members__.items():
                if member_name.lower() == value.lower():
                    return member_obj
            # Try matching by enum value (case-insensitive) e.g. "canny"
            try:
                return cls(
                    value.lower()
                )  # This is the primary way for matching value "canny"
            except ValueError:
                pass  # Value did not match

        valid_options = [m.value for m in cls] + list(cls.__members__.keys())
        raise ValueError(
            f"Invalid filter type: '{value}'. Must be one of (case-insensitive values or names): {valid_options}"
        )


class MatchMethod(str, Enum):
    """Template matching algorithms with validation"""

    CCoeffNormed = "ccoeff_normed"
    SqDiffNormed = "sqdiff_normed"
    CCorrNormed = "ccorr_normed"
    SqDiff = (
        "sqdiff"  # Non-normalized, typically not used with single threshold directly
    )
    CCoeff = "ccoeff"  # Non-normalized
    CCorr = "ccorr"  # Non-normalized

    @classmethod
    def validate(cls, value: Any) -> "MatchMethod":
        """Validate and convert input to MatchMethod"""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            # 1. Try matching by member name (e.g., CCoeffNormed) - allow case variations for robustness
            # Example: "CCoeffNormed", "ccoeffnormed" should map to MatchMethod.CCoeffNormed
            for member_name_in_enum, member_instance in cls.__members__.items():
                if member_name_in_enum.lower() == value.lower():
                    return member_instance

            # 2. Try matching by value (e.g. "ccoeff_normed") from raw string or processed string
            # This handles cases like "cv2.TM_CCOEFF_NORMED" -> "ccoeff_normed" (value of enum)
            # Also handles direct value match like "ccoeff_normed".
            processed_value_for_enum = (
                value.split(".")[-1].strip().lower()
            )  # e.g. "tm_ccoeff_normed" -> "ccoeff_normed"
            try:
                return cls(
                    processed_value_for_enum
                )  # Attempt to init enum from its string value
            except ValueError:
                pass  # Processed value didn't match any enum member's value

        valid_options = [m.value for m in cls] + list(cls.__members__.keys())
        raise ValueError(
            f"Invalid match method: '{value}'. Must be one of (case-insensitive values, names, or prefixed names): {valid_options}"
        )


class TemplateMatchResult:  # This class seems to be a manual Pydantic-like model.
    # Consider if a Pydantic model would be beneficial here for consistency.
    """Immutable container for template matching results"""

    def __init__(
        self,
        success: bool,
        template_path: str,
        score: float,
        threshold: float,
        center_coords: Tuple[int, int],
        top_left_coords: Tuple[int, int],
        size: Tuple[int, int],  # width, height
        offset_coords: Tuple[int, int],  # user-provided offset that was applied
        verify_result: Optional[
            Tuple[bool, float]
        ] = None,  # (confirmed_bool, verify_score_float)
        processing_details: Optional[
            Dict[str, Any]
        ] = None,  # e.g., filter_type, method, canny_params
        frame_info: Optional[Dict[str, Any]] = None,  # e.g., timestamp, processing_time
    ):
        # Basic validation, more detailed could be added (e.g. using ParameterValidator)
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        if not isinstance(template_path, str):
            raise TypeError("template_path must be a string")
        if not isinstance(score, (int, float)):
            raise TypeError("score must be a number")
        # Score range can be [-1, 1] for ccoeff_normed, so 0-1 validation here is too strict
        # if not (0.0 <= score <= 1.0) and not (-1.0 <= score <= 1.0 and "coeff_normed" in str(processing_details)): # Heuristic
        #     logger.warning(f"Score {score} for {template_path} is outside typical ranges, but accepted.")

        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number")
        # Threshold range can also vary based on method.

        def _is_int_tuple_len2(val, name):
            if not (
                isinstance(val, tuple)
                and len(val) == 2
                and all(isinstance(i, int) for i in val)
            ):
                raise ValueError(f"{name} must be a tuple of two integers, got {val}")
            return val

        self.success = success
        self.template_path = template_path
        self.score = float(score)
        self.threshold = float(threshold)
        self.center_coords = _is_int_tuple_len2(center_coords, "center_coords")
        self.top_left_coords = _is_int_tuple_len2(top_left_coords, "top_left_coords")
        self.size = _is_int_tuple_len2(size, "size")
        if not (self.size[0] >= 0 and self.size[1] >= 0):
            raise ValueError("size components must be non-negative")
        self.offset_coords = _is_int_tuple_len2(offset_coords, "offset_coords")

        self.verify_result = verify_result  # Tuple or None
        self.processing_details = (
            processing_details if processing_details is not None else {}
        )
        self.frame_info = frame_info if frame_info is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        data = {
            "success": self.success,
            "template_path": self.template_path,
            "score": self.score,
            "threshold": self.threshold,
            "center_coords": list(self.center_coords),  # List for JSON friendliness
            "top_left_coords": list(self.top_left_coords),
            "size": list(self.size),
            "offset_coords": list(self.offset_coords),
            "processing_details": self.processing_details,
            "frame_info": self.frame_info,
        }
        if self.verify_result:
            data["verify_confirmed"] = self.verify_result[0]
            data["verify_score"] = self.verify_result[1]
        return data


class MJPEGStreamStatus:  # Also a manual Pydantic-like model.
    """Immutable container for MJPEG stream status information"""

    def __init__(
        self,
        active: bool,
        ready: bool,  # Is stream providing frames? (derived from active and frame_age or status_message)
        frame_age: float,  # Seconds since last frame, float('inf') if no frames yet
        resolution: Tuple[int, int],  # width, height (0,0 if unknown)
        status_message: str,
        error_count: int = 0,  # e.g. consecutive decode errors
    ):
        if not isinstance(active, bool):
            raise TypeError("active must be boolean")
        if not isinstance(ready, bool):
            raise TypeError("ready must be boolean")
        if not (isinstance(frame_age, (int, float)) and frame_age >= 0):
            raise ValueError("frame_age must be a non-negative number")
        if not (
            isinstance(resolution, tuple)
            and len(resolution) == 2
            and all(isinstance(x, int) and x >= 0 for x in resolution)
        ):
            raise ValueError("resolution must be a tuple of two non-negative integers")
        if not isinstance(status_message, str):
            raise TypeError("status_message must be a string")
        if not (isinstance(error_count, int) and error_count >= 0):
            raise ValueError("error_count must be a non-negative integer")

        self.active = active
        self.ready = ready
        self.frame_age = frame_age
        self.resolution = resolution
        self.status_message = status_message
        self.error_count = error_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary format"""
        return {
            "active": self.active,
            "ready": self.ready,
            "frame_age_seconds": (
                self.frame_age if self.frame_age != float("inf") else None
            ),  # null for JSON if inf
            "frame_resolution": (
                f"{self.resolution[0]}x{self.resolution[1]}"
                if self.resolution != (0, 0)
                else "N/A"
            ),
            "status_message": self.status_message,
            "error_count": self.error_count,
        }


class TemplateCacheEntry(NamedTuple):
    processed_template: np.ndarray  # The actual image data (e.g., after Canny)
    width: int  # Width of the processed_template
    height: int  # Height of the processed_template
    processing_details: (
        str  # String describing how it was processed (e.g., "canny_50_150")
    )


# Custom Exception Classes
class ImageSearchError(Exception):
    """Base error class for image search operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details if details is not None else {}
        self.error_type = self.__class__.__name__
        # Avoid logging directly in __init__ of exceptions, let handlers do it.
        # logger.error(f"{self.error_type}: {message} | Details: {self.details}")


class ValidationError(ImageSearchError):
    """Error for general validation failures (broader than just parameters)."""


class InvalidParameterError(ValidationError):  # Specific type of ValidationError
    """Error for invalid input parameters to functions or API endpoints."""


class ResourceNotFoundError(ImageSearchError):
    """Error when a required resource (e.g., file) is not found."""


class FilterProcessingError(ImageSearchError):
    """Error during image filter processing (e.g., Canny failure)."""


class TemplateLoadError(ImageSearchError):
    """Error during template image loading, decoding, or initial processing for cache."""


class TemplateMatchError(ImageSearchError):
    """Error during the core template matching process (cv2.matchTemplate, result interpretation)."""


class MJPEGStreamError(ImageSearchError):
    """Error related to MJPEG stream operations (connection, reading, decoding frames from stream)."""


class CacheError(ImageSearchError):
    """Error related to cache operations (e.g., TemplateCache, INICache)."""


# Parameter Validation Class
class ParameterValidator:
    """Utility class for validating various parameters."""

    @staticmethod
    def validate_threshold(value: Any) -> float:
        """Validates the threshold parameter."""
        if not isinstance(value, (int, float)):
            raise InvalidParameterError(
                f"Threshold must be a number, got {type(value).__name__} '{value}'."
            )
        # Threshold for ccoeff_normed can be negative.
        # For sqdiff_normed, it's typically [0,1].
        # Let's assume a broad valid range, specific checks might be needed based on method.
        # If we strictly enforce [0,1] here, it limits ccoeff_normed.
        # For now, keeping the 0-1 validation as it was common, but with a note.
        # Consider making threshold validation method-dependent or relaxing this.
        if not -1.0 <= float(value) <= 1.0:  # Allow range for ccoeff_normed
            logger.warning(
                f"Threshold {value} is outside the typical [0,1] or [-1,1] range, but accepted."
            )
        # Original strict check:
        # if not 0.0 <= float(value) <= 1.0:
        #     raise InvalidParameterError(
        #         f"Threshold must be between 0.0 and 1.0, got {value}."
        #     )
        return float(value)

    @staticmethod
    def validate_coordinates(
        x: Optional[Any],  # Changed to Any to catch type errors better
        y: Optional[Any],
        image_width: int,
        image_height: int,
        param_name_x: str = "x",
        param_name_y: str = "y",
    ) -> Tuple[int, int]:
        """
        Validates a 2D coordinate (x, y) against image boundaries.
        Raises InvalidParameterError if coordinates are invalid.
        """
        if x is None:
            raise InvalidParameterError(
                f"Coordinate '{param_name_x}' must be provided."
            )
        if y is None:
            raise InvalidParameterError(
                f"Coordinate '{param_name_y}' must be provided."
            )

        if not isinstance(x, int):
            raise InvalidParameterError(
                f"Coordinate '{param_name_x}' ('{x}') must be an integer."
            )
        if not isinstance(y, int):
            raise InvalidParameterError(
                f"Coordinate '{param_name_y}' ('{y}') must be an integer."
            )

        if image_width <= 0 or image_height <= 0:
            raise InvalidParameterError(
                f"Image dimensions ({image_width}x{image_height}) must be positive for coordinate validation."
            )

        if not (
            0 <= x < image_width
        ):  # Coordinates are 0-indexed, so x must be < width
            raise InvalidParameterError(
                f"Coordinate '{param_name_x}' ({x}) is out of bounds for image width {image_width}. Must be in [0, {image_width - 1}]."
            )
        if not (0 <= y < image_height):  # y must be < height
            raise InvalidParameterError(
                f"Coordinate '{param_name_y}' ({y}) is out of bounds for image height {image_height}. Must be in [0, {image_height - 1}]."
            )
        return x, y

    @staticmethod
    def validate_search_region(
        x1: Optional[Any],
        y1: Optional[Any],
        x2: Optional[Any],
        y2: Optional[Any],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, int]:
        """Validates search region parameters. Returns a dict {'x1', 'y1', 'x2', 'y2'}."""
        all_none = all(v is None for v in (x1, y1, x2, y2))
        any_none = any(v is None for v in (x1, y1, x2, y2))

        if all_none:
            if frame_width <= 0 or frame_height <= 0:
                raise InvalidParameterError(
                    f"Frame dimensions ({frame_width}x{frame_height}) must be positive when defaulting search region."
                )
            return {"x1": 0, "y1": 0, "x2": frame_width, "y2": frame_height}

        if any_none and not all_none:  # Mix of None and provided values
            raise InvalidParameterError(
                "Search region parameters (x1, y1, x2, y2) must all be provided or all be omitted."
            )

        # At this point, none of x1, y1, x2, y2 are None. Proceed with type and value checks.
        coord_params = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        int_coords: Dict[str, int] = {}
        for name, val in coord_params.items():
            if not isinstance(val, int):
                raise InvalidParameterError(
                    f"Search region parameter '{name}' ('{val}') must be an integer."
                )
            if val < 0:  # x1,y1,x2,y2 cannot be negative
                raise InvalidParameterError(
                    f"Search region parameter '{name}' ({val}) cannot be negative."
                )
            int_coords[name] = val

        ix1, iy1, ix2, iy2 = (
            int_coords["x1"],
            int_coords["y1"],
            int_coords["x2"],
            int_coords["y2"],
        )

        if ix1 >= ix2 or iy1 >= iy2:
            raise InvalidParameterError(
                f"Search region coordinates must satisfy x1 < x2 and y1 < y2. Got ({ix1},{iy1})-({ix2},{iy2})."
            )

        # x2 and y2 are exclusive upper bounds for region, but represent coordinates *within* frame dimensions
        if (
            ix2 > frame_width or iy2 > frame_height
        ):  # x2 is width of region, y2 is height of region
            raise InvalidParameterError(
                f"Search region ({ix1},{iy1})-({ix2},{iy2}) exceeds frame dimensions ({frame_width}x{frame_height}). x2 must be <= frame_width and y2 <= frame_height."
            )
        return {"x1": ix1, "y1": iy1, "x2": ix2, "y2": iy2}

    @staticmethod
    def validate_canny_params(t1: Optional[Any], t2: Optional[Any]) -> Tuple[int, int]:
        """Validates Canny edge detection thresholds."""
        from modules.constants import (  # Local import for defaults
            DEFAULT_CANNY_T1,
            DEFAULT_CANNY_T2,
        )

        if t1 is None and t2 is None:
            return DEFAULT_CANNY_T1, DEFAULT_CANNY_T2

        if t1 is None or t2 is None:  # Only one is provided
            raise InvalidParameterError(
                "Canny thresholds t1 and t2 must both be provided or both be omitted."
            )

        if not isinstance(t1, int) or not isinstance(t2, int):
            raise InvalidParameterError(
                f"Canny thresholds t1 ('{t1}') and t2 ('{t2}') must be integers."
            )

        if not (
            0 <= t1 <= 255 and 0 <= t2 <= 255
        ):  # OpenCV Canny thresholds are typically 0-255
            raise InvalidParameterError(
                f"Canny thresholds t1 ({t1}) and t2 ({t2}) must be between 0 and 255."
            )
        if (
            t1 >= t2
        ):  # Standard Canny constraint: lower threshold must be less than upper threshold
            raise InvalidParameterError(
                f"Canny threshold t1 ({t1}) must be strictly less than t2 ({t2})."
            )
        return t1, t2

    @staticmethod
    def validate_image_format(image_path: str) -> str:
        """Validates if the image format is supported by checking extension. Returns normalized path."""
        if not isinstance(image_path, str):
            raise InvalidParameterError(
                f"Image path must be a string, got {type(image_path).__name__}."
            )

        if not image_path.strip():
            raise InvalidParameterError("Image path cannot be empty.")

        # Supported formats (lowercase extensions)
        # OpenCV's imread supports a wide range, these are common ones.
        valid_formats = [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
            ".webp",
            ".jp2",
        ]

        try:
            ext = os.path.splitext(image_path)[1].lower()
            if not ext:
                raise InvalidParameterError(
                    f"Image path '{image_path}' has no file extension."
                )
            if ext not in valid_formats:
                raise InvalidParameterError(
                    f"Unsupported image format extension: '{ext}'. Supported: {', '.join(valid_formats)}."
                )
        except (
            Exception
        ) as e:  # Catch potential errors from os.path.splitext if path is weird
            raise InvalidParameterError(
                f"Invalid image path for format validation: '{image_path}'. Error: {e}"
            )

        return os.path.normpath(image_path)  # Return normalized path

    @staticmethod
    def validate_mjpeg_url(url: Optional[str]) -> str:
        """Validates the MJPEG URL format."""
        if not url or not isinstance(url, str) or not url.strip():
            raise InvalidParameterError(
                f"MJPEG URL must be a non-empty string, got: '{url}'"
            )

        # Basic check for common schemes. Add more as needed (e.g. file:// for local testing if supported)
        # Common library like `urllib.parse` can be used for more robust URL validation.
        parsed_url = None
        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(url)
        except ImportError:  # Should not happen in modern Python
            pass

        if parsed_url:
            if parsed_url.scheme not in ["http", "https", "rtsp"]:  # Added rtsp
                raise InvalidParameterError(
                    f"MJPEG URL '{url}' scheme '{parsed_url.scheme}' is not supported. Must be http, https, or rtsp."
                )
            if not parsed_url.netloc:  # Must have a host part
                raise InvalidParameterError(
                    f"MJPEG URL '{url}' is missing a valid network location (host/domain)."
                )
        else:  # Fallback if urlparse fails or not used
            if not (
                url.startswith("http://")
                or url.startswith("https://")
                or url.startswith("rtsp://")
            ):
                raise InvalidParameterError(
                    f"MJPEG URL '{url}' must start with http://, https://, or rtsp://."
                )
        return url.strip()  # Return stripped URL
