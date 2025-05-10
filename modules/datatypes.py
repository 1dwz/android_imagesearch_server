# -*- coding: utf-8 -*-
"""
Custom data types and validation utilities for image search server.
"""
import logging
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple, TypedDict

import numpy as np  # Import numpy for array type hinting

logger = logging.getLogger(__name__)


# Define TypedDict for FrameDataCache data structure
class FrameDataDict(TypedDict):
    """Typed dictionary for the data stored in FrameDataCache."""

    frame: Optional[Any]  # Use Any for now to avoid strict numpy type hint issues
    gray_frame: Optional[Any]  # Use Any for now
    shape: Optional[Tuple[int, int]]
    timestamp: float
    frame_number: int
    status: str


# Define NamedTuple for MJPEGStreamManager status info
class MJPEGStatusInfo(NamedTuple):
    """Named tuple for MJPEG stream status information."""

    is_active: bool
    status_message: str
    last_frame_timestamp: float


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


# New: Custom exception for filter processing errors
class FilterProcessingError(Exception):
    """Custom exception for filter processing errors"""

    pass  # This can remain simple


# New: NamedTuple for Template Cache Entry
class TemplateCacheEntry(NamedTuple):
    """Data structure for an entry in the TemplateCache."""

    processed_template: (
        np.ndarray
    )  # The processed template image (e.g., grayscale, filtered)
    width: int  # Width of the processed template
    height: int  # Height of the processed template
    processing_details: str  # String describing the processing applied (e.g., "Filter: none", "Filter: canny (Params: ...)").
