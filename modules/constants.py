# -*- coding: utf-8 -*-
"""
Constants module for the Image Matching API Server
Contains all constant values used throughout the application
"""

# Version Constants
SERVER_VERSION = "6.0"

# Server Defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 60003
DEFAULT_MJPEG_URL = "http://192.168.1.3:8080/stream.mjpeg"

# Timeout Settings
DEFAULT_IDLE_TIMEOUT = 30.0
DEFAULT_WAKEUP_TIMEOUT = 15.0

# Image Processing Defaults
DEFAULT_FILTER_TYPE = "none"
DEFAULT_MATCH_METHOD = "ccoeff_normed"
DEFAULT_THRESHOLD = 0.8
DEFAULT_CANNY_T1 = 50
DEFAULT_CANNY_T2 = 150
DEFAULT_FILTER_PARAM_CANNY_T1 = 50

# Cache Settings
DEFAULT_TEMPLATE_CACHE_SIZE = 1000
DEFAULT_INI_CACHE_SIZE = 100
CACHE_KEY_HASH_LENGTH = 8

# Debug Settings
DEFAULT_DEBUG_SAVE_DIR = "debug_images"
DEFAULT_MAX_DEBUG_FILES = 100

# Match Methods
MATCH_METHODS = ["ccoeff_normed", "sqdiff_normed", "ccorr_normed"]

# Filter Types
FILTER_TYPES = ["none", "canny"]

# OpenCV Match Method Mappings
CV2_MATCH_METHODS = {
    "ccoeff_normed": "cv2.TM_CCOEFF_NORMED",
    "sqdiff_normed": "cv2.TM_SQDIFF_NORMED",
    "ccorr_normed": "cv2.TM_CCORR_NORMED",
}

# HTTP Response Fields
RESPONSE_FIELDS = [
    "found",
    "center_x",
    "center_y",
    "template_name",
    "template_path",
    "score",
    "top_left_x",
    "top_left_y",
    "width",
    "height",
    "top_left_x_with_offset",
    "top_left_y_with_offset",
    "offset_applied_x",
    "offset_applied_y",
    "verify_wait",
    "verify_confirmed",
    "verify_score",
    "recheck_status",
    "recheck_frame_timestamp",
    "search_region_x1",
    "search_region_y1",
    "search_region_x2",
    "search_region_y2",
    "search_region_full_search",
    "filter_type_used",
    "match_method_used",
    "frame_timestamp",
    "frame_width",
    "frame_height",
    "threshold",
    "highest_score",
    "error",
]
