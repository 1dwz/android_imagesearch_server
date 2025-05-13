import logging

import cv2

# Logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Server
SERVER_VERSION = "6.0"  # Defined in __init__.py, but useful here too for clarity.
DEFAULT_PORT = 8000
DEFAULT_MJPEG_URL = "http://localhost:8080/stream"  # Placeholder, should be configured
DEFAULT_SERVER_CONFIG_PATH = "config/server.ini"  # Relative to project root or CWD
LOGS_DIR = "logs"  # Used by manager.py
SERVER_LOG_FILENAME_FORMAT = "server_{port}.log"  # Used by manager.py


# MJPEG Stream
MAX_CONSECUTIVE_EMPTY_CHUNKS = 5
MAX_CONSECUTIVE_DECODE_ERRORS = 10
MAX_MJPEG_BUFFER_SIZE = (
    10 * 1024 * 1024
)  # 10MB - Max size of the bytearray buffer for accumulating MJPEG chunks.
MJPEG_BUFFER_KEEP_SIZE_ON_TRIM = (
    2 * 1024 * 1024
)  # 2MB - When MAX_MJPEG_BUFFER_SIZE is exceeded, trim buffer from start, keeping this many bytes at the end.
# CRITICAL: This value MUST be large enough to typically hold at least one full JPEG frame
# plus some boundary/header data to avoid corrupting the start of the next frame search.
# If frames are very large (e.g. >2MB), this might need adjustment, or MAX_MJPEG_BUFFER_SIZE increased.
DEFAULT_IDLE_TIMEOUT = 300.0  # Seconds - if no activity (frames received or get_frame calls), stream stops.
DEFAULT_WAKEUP_TIMEOUT = (
    15.0  # Seconds - timeout for stream to provide first frame after starting.
)


# Image Processing
DEFAULT_FILTER_TYPE = "none"
DEFAULT_MATCH_METHOD = "ccoeff_normed"
DEFAULT_THRESHOLD = 0.8
DEFAULT_CANNY_T1 = (
    100  # Default Canny lower threshold if not specified in config or params
)
DEFAULT_CANNY_T2 = (
    200  # Default Canny upper threshold if not specified in config or params
)
SUPPORTED_FILTER_TYPES = [
    "none",
    "canny",
]  # Used by FilterType enum validation, keep in sync
SUPPORTED_MATCH_METHODS = [  # Used by MatchMethod enum validation, keep in sync
    "ccoeff_normed",
    "sqdiff_normed",
    "ccorr_normed",  # Added ccorr_normed as it's a common one
    "sqdiff",
    "ccoeff",
    "ccorr",
    # The following seem to be duplicates or older naming conventions from constants.py history
    # "tmpl_match_method_sqdiff",
    # "tmpl_match_method_sqdiff_normed",
    # "tmpl_match_method_ccoeff",
    # "tmpl_match_method_ccoeff_normed",
    # "tmpl_match_method_ccorr",
    # "tmpl_match_method_ccorr_normed",
]

CV2_MATCH_METHODS_MAP = {
    "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
    "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
    "ccorr_normed": cv2.TM_CCORR_NORMED,  # Added ccorr_normed
    "sqdiff": cv2.TM_SQDIFF,
    "ccoeff": cv2.TM_CCOEFF,
    "ccorr": cv2.TM_CCORR,
    # Ensure this map covers all values in MatchMethod enum and SUPPORTED_MATCH_METHODS
    # The longer "tmpl_match_method_..." keys were removed for consistency with MatchMethod enum.
    # If those longer keys are still expected from some input source (e.g. old INI files),
    # then MatchMethod.validate needs to handle them or they need to be added back here and to enum.
    # For now, assuming shorter names are canonical.
}


# Template Cache (ImageProcessor)
CACHE_KEY_HASH_LENGTH = 16  # Length of SHA256 hash used for template cache keys.
DEFAULT_TEMPLATE_CACHE_SIZE = (
    1000  # Default max items in ImageProcessor's template cache.
)
# DEFAULT_TEMPLATE_CACHE_TTL = 3600  # Seconds - TTL Not currently implemented in TemplateCache

# INI Cache (modules.utils.INICache) - utils.py has its own default (100)
# If this needs to be globally configurable via server.ini, INICache init needs adjustment.
# DEFAULT_INI_CACHE_SIZE = 100 # Example if it were to be centralized

# FrameDataCache (internal to MJPEGStreamReader, no separate constants needed here unless for defaults it consumes)
# DEFAULT_FRAME_CACHE_SIZE = 1 # (Effectively, as it stores one latest frame)
# DEFAULT_FRAME_CACHE_TTL = 1.0 # (Not directly applicable, frame_age is dynamic)

# API
API_PREFIX = "/api/v1"  # If a global prefix is used for all routes
API_TITLE = "Image Search Server API"
# API_VERSION = "1.0.0" # Potentially sync with SERVER_VERSION or manage separately for API contract versioning

# Manager
DEFAULT_MULTI_CONFIG_PATH = "multi_config.json"  # For manager.py
PROCESS_START_TIMEOUT = 15.0  # seconds - For manager.py, waiting for instance to start
REQUEST_TIMEOUT = 5.0  # seconds - For manager.py, default timeout for HTTP requests to instances (was 2.0, increased for potentially slower internal APIs)

# Statuses (used by manager.py and potentially API responses)
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_UNRESPONSIVE = "unresponsive"
STATUS_ERROR = "error"
# STATUS_NOT_PERFORMED = "Not performed" # More specific to test results, perhaps not general status
# STATUS_WAITING = "Waiting"
# STATUS_PASSED = "Passed"
# STATUS_FAILED = "Failed"

# Other
INFINITE_TIMEOUT = float("inf")  # Used in mjpeg.py health_check for frame_age
