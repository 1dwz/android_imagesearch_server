# -*- coding: utf-8 -*-
"""
API Module
Defines FastAPI endpoints and request handlers for the image matching service
"""
import logging
import os
import time
from pathlib import Path  # Import Path for path operations
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic import validator as pydantic_validator  # aliased to avoid conflict

from modules.app_config import AppConfig  # For create_app signature
from modules.constants import (  # Ensure this is up-to-date and API_TITLE imported
    API_TITLE,
    SERVER_VERSION,
)
from modules.datatypes import FilterType  # For validation in batch search (example)
from modules.datatypes import MatchMethod  # For validation in batch search (example)
from modules.datatypes import TemplateMatchResult  # For type hinting detailed result
from modules.datatypes import (
    ImageSearchError,
    InvalidParameterError,
    MJPEGStreamError,
    ParameterValidator,
    ResourceNotFoundError,
    ValidationError,
)
from modules.image_processor import ImageProcessor
from modules.mjpeg import MJPEGStreamReader
from modules.utils import (
    parse_ini_file,
    save_debug_frame,
)

# FrameDataCache is internal to MJPEGStreamReader, so no direct import needed here.


logger = logging.getLogger(__name__)


# Pydantic Models
class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall server health (ok, warning, error)")
    message: str = Field(..., description="Human-readable status message")
    server_version: str = Field(..., description="Server software version")
    mjpeg_stream_health: Dict[str, Any] = Field(
        ..., description="Detailed MJPEG stream health"
    )


class MJPEGURLUpdateRequest(BaseModel):
    new_url: str = Field(..., description="The new MJPEG URL to use for the stream.")

    @pydantic_validator("new_url")
    def validate_new_url_format(cls, value):  # Renamed for clarity
        try:
            return ParameterValidator.validate_mjpeg_url(value)
        except (
            InvalidParameterError
        ) as e:  # ParameterValidator raises InvalidParameterError
            raise ValueError(
                str(e)
            )  # Pydantic validators expect ValueError, TypeError, or AssertionError


class SearchParamsModel(BaseModel):  # For query parameters in /search
    img: str = Field(..., description="Absolute path to the template image file.")
    filter_type: Optional[str] = Field(
        None, description="Filter type ('none' or 'canny')."
    )
    match_method: Optional[str] = Field(None, description="Template matching method.")
    threshold: Optional[float] = Field(
        None,
        description="Matching threshold (e.g., 0.0 to 1.0, or -1.0 to 1.0 for ccoeff_normed).",
    )
    x1: Optional[int] = Field(None, description="Search region top-left X coordinate.")
    y1: Optional[int] = Field(None, description="Search region top-left Y coordinate.")
    x2: Optional[int] = Field(
        None, description="Search region bottom-right X coordinate."
    )
    y2: Optional[int] = Field(
        None, description="Search region bottom-right Y coordinate."
    )
    offsetx: Optional[int] = Field(
        0, description="X offset to add to result coordinates."  # Default to 0
    )
    offsety: Optional[int] = Field(
        0, description="Y offset to add to result coordinates."  # Default to 0
    )
    canny_t1: Optional[int] = Field(
        None, description="Canny edge detection lower threshold."
    )
    canny_t2: Optional[int] = Field(
        None, description="Canny edge detection upper threshold."
    )

    @pydantic_validator("img")
    def validate_img_path_is_absolute(cls, v: str) -> str:
        if not os.path.isabs(v):
            raise ValueError("Template image path 'img' must be an absolute path.")
        # Further validation (e.g., existence) can be done by the image_processor or a shared utility
        # For example, ParameterValidator.validate_image_format(v) could be called here too,
        # but it's also called in image_processor._load_template.
        # To avoid double validation error messages, decide where it's best handled.
        # Keeping it simple here, ImageProcessor will handle detailed file validation.
        return os.path.normpath(v)


class BatchSearchRequestItem(BaseModel):
    template_path: str  # Should also be absolute, consider adding validator or documenting requirement
    # Optional per-template overrides
    filter_type: Optional[str] = None
    match_method: Optional[str] = None
    threshold: Optional[float] = None
    search_region: Optional[Dict[str, int]] = (
        None  # e.g. {"x1":0, "y1":0, "x2":100, "y2":100}
    )
    offset: Optional[Dict[str, int]] = None  # e.g. {"x":0, "y":0}
    canny_params: Optional[Dict[str, int]] = None  # e.g. {"t1":50, "t2":150}

    @pydantic_validator("template_path")
    def validate_template_path_abs(cls, v: str) -> str:
        if not os.path.isabs(v):
            raise ValueError("Batch item 'template_path' must be an absolute path.")
        return os.path.normpath(v)

    # Example: Add validators for filter_type, match_method if specific values are desired beyond string
    @pydantic_validator("filter_type")
    def validate_filter_type_str(cls, v):
        if v is None:
            return v
        try:
            FilterType.validate(v)  # Use the enum's validator
        except ValueError as e:
            raise ValueError(f"Invalid filter_type in batch item: {e}")
        return v

    @pydantic_validator("match_method")
    def validate_match_method_str(cls, v):
        if v is None:
            return v
        try:
            MatchMethod.validate(v)  # Use the enum's validator
        except ValueError as e:
            raise ValueError(f"Invalid match_method in batch item: {e}")
        return v


class BatchSearchRequest(BaseModel):
    templates: List[BatchSearchRequestItem] = Field(
        ..., min_items=1, description="List of templates to search for."
    )
    # Global defaults for the batch, can be overridden by individual items
    filter_type: Optional[str] = Field(
        None, description="Global filter type if not set in item."
    )
    match_method: Optional[str] = Field(
        None, description="Global match method if not set in item."
    )
    threshold: Optional[float] = Field(
        None, description="Global threshold if not set in item."
    )

    @pydantic_validator("filter_type")
    def validate_global_filter_type_str(cls, v):
        if v is None:
            return v
        try:
            FilterType.validate(v)
        except ValueError as e:
            raise ValueError(f"Invalid global filter_type: {e}")
        return v

    @pydantic_validator("match_method")
    def validate_global_match_method_str(cls, v):
        if v is None:
            return v
        try:
            MatchMethod.validate(v)
        except ValueError as e:
            raise ValueError(f"Invalid global match_method: {e}")
        return v


class ErrorResponse(BaseModel):
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None


# FastAPI App Creation
def create_app(app_config: AppConfig) -> FastAPI:
    """Creates and new_image_processor_configures the FastAPI application instance."""

    # Initialize FastAPI app
    app = FastAPI(
        title=API_TITLE,
        description="API for template-based image matching in MJPEG streams.",
        version=SERVER_VERSION,
    )

    app.state.app_config = app_config  # Store AppConfig for access

    # Server settings from AppConfig (this gets the whole server_config dict from main_instance.py)
    server_settings_dict: Dict[str, Any] = app_config.get_image_processor_config()
    app.state.server_settings = server_settings_dict

    # Initialize MJPEGStreamReader
    stream_config_params = app_config.get_stream_config()
    try:
        app.state.stream_reader = MJPEGStreamReader(
            mjpeg_url=stream_config_params["mjpeg_url"],
            idle_timeout=stream_config_params.get(
                "idle_timeout"
            ),  # .get() handles missing keys from AppConfig source
            wakeup_timeout=stream_config_params.get("wakeup_timeout"),
        )
        if not app.state.stream_reader.start():
            logger.warning(
                f"MJPEG stream {stream_config_params['mjpeg_url']} failed to start automatically on app initialization."
            )
        else:
            logger.info(
                f"MJPEG stream {stream_config_params['mjpeg_url']} started successfully on app initialization."
            )
    except MJPEGStreamError as e:
        logger.error(
            f"Failed to initialize MJPEGStreamReader for {stream_config_params.get('mjpeg_url', 'N/A')}: {e}. Stream will be unavailable."
        )
        app.state.stream_reader = None
    except Exception as e_init_mjpeg:  # Catch any other init error
        logger.error(
            f"Unexpected error initializing MJPEGStreamReader for {stream_config_params.get('mjpeg_url', 'N/A')}: {e_init_mjpeg}. Stream will be unavailable."
        )
        app.state.stream_reader = None

    # Initialize ImageProcessor
    try:
        app.state.image_processor = ImageProcessor(server_settings_dict)
    except Exception as e:
        logger.error(
            f"Failed to initialize ImageProcessor: {e}. Search endpoints will likely fail."
        )
        app.state.image_processor = None

    # Debug image saving setup
    if server_settings_dict.get(
        "enable_debug_saving", False
    ) and server_settings_dict.get("debug_save_dir"):
        try:
            os.makedirs(server_settings_dict["debug_save_dir"], exist_ok=True)
            logger.info(
                f"Debug image saving enabled. Directory: {server_settings_dict['debug_save_dir']}"
            )
        except OSError as e:
            logger.error(
                f"Failed to create debug_save_dir '{server_settings_dict['debug_save_dir']}': {e}"
            )
            server_settings_dict["enable_debug_saving"] = (
                False  # Disable if dir creation fails
            )

    # Exception Handlers
    @app.exception_handler(ImageSearchError)
    async def image_search_error_handler(request: Request, exc: ImageSearchError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(exc, (ValidationError, InvalidParameterError)):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(
            exc, ResourceNotFoundError
        ):  # Covers template file not found from ImageProcessor/parse_ini_file
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, MJPEGStreamError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        # Log more context if available from request (e.g. client IP)
        logger.warning(
            f"ImageSearchError caught: Type='{exc.error_type}', Msg='{str(exc)}', Details='{exc.details}', Path='{request.url.path}', Client='{request.client.host if request.client else 'N/A'}'"
        )
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=str(exc), error_type=exc.error_type, details=exc.details
            ).dict(
                exclude_none=True
            ),  # Exclude details if None
        )

    @app.exception_handler(HTTPException)  # Handles FastAPI's own HTTPExceptions
    async def http_exception_handler_override(request: Request, exc: HTTPException):
        logger.info(
            f"FastAPI HTTPException: Status={exc.status_code}, Detail='{exc.detail}', Path='{request.url.path}', Client='{request.client.host if request.client else 'N/A'}'"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=str(exc.detail),
                error_type="HTTPException",  # Generic type for FastAPI's own
            ).dict(exclude_none=True),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)  # Generic fallback for unhandled errors
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception(  # Use logger.exception to include stack trace for unexpected errors
            f"Unhandled generic exception during request to {request.url.path} by client {request.client.host if request.client else 'N/A'}"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="An unexpected internal server error occurred.",
                error_type=type(exc).__name__,
            ).dict(exclude_none=True),
        )

    # API Endpoints
    @app.get("/health", response_model=HealthResponse)
    async def health_check_endpoint():
        stream_reader: Optional[MJPEGStreamReader] = app.state.stream_reader
        app_cfg: Optional[AppConfig] = app.state.app_config

        mjpeg_url_for_health = "N/A"
        if app_cfg:
            mjpeg_url_for_health = app_cfg.mjpeg_url
        elif (
            stream_reader
        ):  # Fallback if app_config somehow not set but stream_reader is
            mjpeg_url_for_health = stream_reader.mjpeg_url

        if not stream_reader:
            return HealthResponse(
                status="error",
                message="MJPEG stream reader not initialized.",
                server_version=SERVER_VERSION,
                mjpeg_stream_health={
                    "mjpeg_url": mjpeg_url_for_health,
                    "is_active": False,
                    "stream_status_message": "Not Initialized",
                },
            )

        stream_health_data = stream_reader.health_check()

        overall_status = "ok"
        overall_message = "Server and stream are operational."

        if (
            not stream_health_data.get("is_active", False)
            or "Failed" in stream_health_data.get("stream_status_message", "")
            or "Error" in stream_health_data.get("stream_status_message", "")
        ):
            overall_status = "error"
            overall_message = f"MJPEG stream issue: {stream_health_data.get('stream_status_message', 'Unknown stream error')}"
        elif (
            stream_health_data.get("frame_age_seconds", float("inf"))
            > stream_reader.wakeup_timeout
        ):
            overall_status = "warning"
            overall_message = f"Frame age ({stream_health_data.get('frame_age_seconds', 0):.1f}s) exceeds wakeup timeout ({stream_reader.wakeup_timeout}s)."

        return HealthResponse(
            status=overall_status,
            message=overall_message,
            server_version=SERVER_VERSION,
            mjpeg_stream_health=stream_health_data,
        )

    async def _perform_single_search(
        template_path: str,
        search_params_dict: Dict[str, Any],
        response_obj: Optional[
            Response
        ] = None,  # Make Response optional as it's for headers
    ) -> Dict[str, Any]:  # Should map to TemplateMatchResult structure if possible
        """Helper to perform a single template match. Raises ImageSearchError on failure."""
        start_time_search = time.time()

        stream_reader: Optional[MJPEGStreamReader] = app.state.stream_reader
        image_processor: Optional[ImageProcessor] = app.state.image_processor
        server_settings_dict: Dict[str, Any] = (
            app.state.server_settings
        )  # This is the parsed server.ini

        if not stream_reader or not image_processor:
            # This check should ideally be done by the caller or a middleware if critical
            raise MJPEGStreamError(  # More specific than HTTPException for internal logic
                "Server components (stream reader or image processor) not ready."
            )

        # Get frame
        # MJPEGStreamReader.get_frame() attempts to start stream if not active.
        # It returns None if start fails or no frame within timeout.
        frame_data_dict = stream_reader.get_frame()
        if not frame_data_dict or frame_data_dict.get("frame") is None:
            raise MJPEGStreamError(
                "Failed to get a valid frame from MJPEG stream for search. Stream might be down or initializing.",
                details=stream_reader.health_check(),  # Add current stream health
            )

        current_frame = frame_data_dict["frame"]
        frame_timestamp = frame_data_dict["timestamp"]

        # Debug save (if enabled in server_settings_dict)
        if server_settings_dict.get("enable_debug_saving"):
            save_debug_frame(
                current_frame,
                server_settings_dict.get(
                    "debug_save_dir", "debug_images"
                ),  # Default dir if not in config
                prefix=f"search_{Path(template_path).stem}",  # Use Path for robust stem extraction
                max_files=int(server_settings_dict.get("max_debug_files", 100)),
            )

        # Perform matching (ImageProcessor.match_template raises specific ImageSearchErrors)
        match_result_dict = image_processor.match_template(
            current_frame, template_path, search_params_dict
        )

        # Add timing and frame info to response headers and result body
        processing_time_ms = (time.time() - start_time_search) * 1000
        if response_obj:
            response_obj.headers["X-Processing-Time-Ms"] = f"{processing_time_ms:.2f}"
            response_obj.headers["X-Frame-Timestamp"] = (
                f"{frame_timestamp:.3f}"  # Epoch
            )

        # Ensure match_result_dict (from image_processor) is a dict and add frame info.
        # ImageProcessor now returns a dict that should align with TemplateMatchResult structure.
        match_result_dict.setdefault("frame_info", {})  # Ensure key exists
        match_result_dict["frame_info"]["timestamp_epoch"] = frame_timestamp
        match_result_dict["frame_info"][
            "search_processing_time_ms"
        ] = processing_time_ms  # More specific key

        # Convert to TemplateMatchResult.to_dict() for consistent output if needed,
        # or ensure image_processor.match_template already returns this structure.
        # For now, assuming match_result_dict is already in the desired output format.
        return match_result_dict

    @app.get(
        "/search", summary="Search for a single template image via query parameters."
    )
    async def search_endpoint(
        response: Response, params: SearchParamsModel = Query(...)
    ):
        # Pydantic model `params` has validated inputs.
        # `img` path is absolute. `offsetx`, `offsety` have defaults.

        # Convert Pydantic model to dict for image_processor.
        # image_processor.match_template expects Nones where user didn't provide,
        # and it will apply its own defaults (from server.ini or constants).
        search_params_for_processor = params.dict(exclude_none=True)
        # Pydantic's exclude_none=True is good.
        # However, our SearchParamsModel defaults offsetx/offsety to 0.
        # If user omits them, they will be 0. If they are explicitly None in query (not typical for GET),
        # then Pydantic would make them None if Optional typehint and no default.
        # The current SearchParamsModel defaults offsetx/y to 0, so they will always be present.

        # If SearchParamsModel had `offsetx: Optional[int] = None`, then:
        # search_params_for_processor = params.dict() # Keep Nones
        # And ImageProcessor.match_template would handle `params.get("offsetx", 0)`

        # With current SearchParamsModel (offsetx: Optional[int] = 0),
        # params.offsetx will be 0 if not provided.
        # So, params.dict(exclude_none=True) is fine but won't exclude offsetx/y if they are 0.
        # This is okay as ImageProcessor will use them.

        # `img` key will be present.
        template_image_path = search_params_for_processor.pop("img")

        return await _perform_single_search(
            template_image_path, search_params_for_processor, response
        )

    @app.get("/search_ini", summary="Search for a template defined in an INI file.")
    async def search_ini_endpoint(
        response: Response,
        ini_path: str = Query(
            ..., description="Absolute path to the INI configuration file."
        ),
    ):
        # Validate ini_path is absolute
        if not os.path.isabs(ini_path):
            raise InvalidParameterError(  # Or use HTTPException directly
                f"INI path '{ini_path}' must be an absolute path."
            )
        resolved_ini_path = os.path.normpath(ini_path)

        # Parse INI (parse_ini_file raises FileNotFoundError, ValueError)
        # It resolves template_path inside INI to absolute relative to INI's dir.
        try:
            ini_params = parse_ini_file(resolved_ini_path)
        except FileNotFoundError as e:
            raise ResourceNotFoundError(
                str(e), details={"ini_path": resolved_ini_path}
            ) from e
        except ValueError as e:  # Covers malformed INI or missing sections/keys
            raise InvalidParameterError(
                f"Error parsing INI file '{resolved_ini_path}': {e}"
            ) from e

        template_path_from_ini = ini_params.pop("template_path")  # This is now absolute

        # Map INI keys to image_processor expected param names
        # e.g., match_range_x1 -> x1, offset_x -> offsetx
        processor_params: Dict[str, Any] = {}
        if "filter_type" in ini_params:
            processor_params["filter_type"] = ini_params["filter_type"]
        if "match_method" in ini_params:
            processor_params["match_method"] = ini_params["match_method"]
        if "threshold" in ini_params:
            processor_params["threshold"] = ini_params["threshold"]
        if "match_range_x1" in ini_params:
            processor_params["x1"] = ini_params["match_range_x1"]
        if "match_range_y1" in ini_params:
            processor_params["y1"] = ini_params["match_range_y1"]
        if "match_range_x2" in ini_params:
            processor_params["x2"] = ini_params["match_range_x2"]
        if "match_range_y2" in ini_params:
            processor_params["y2"] = ini_params["match_range_y2"]
        processor_params["offsetx"] = ini_params.get(
            "offset_x", 0
        )  # Default offset to 0
        processor_params["offsety"] = ini_params.get("offset_y", 0)
        if "canny_t1" in ini_params:
            processor_params["canny_t1"] = ini_params["canny_t1"]
        if "canny_t2" in ini_params:
            processor_params["canny_t2"] = ini_params["canny_t2"]

        # `waitforrecheck` from INI is not directly used by `_perform_single_search`.
        # If it's for client-side logic, it can be returned.
        # For now, it's ignored by the core search.

        return await _perform_single_search(
            template_path_from_ini, processor_params, response
        )

    @app.post(
        "/batch_search", summary="Search for multiple templates in a single request."
    )
    async def batch_search_endpoint(
        request_body: BatchSearchRequest,
        response: Response,  # FastAPI handles request_body validation
    ):
        results = []
        stream_reader: Optional[MJPEGStreamReader] = app.state.stream_reader
        image_processor: Optional[ImageProcessor] = app.state.image_processor
        server_settings_dict: Dict[str, Any] = app.state.server_settings

        if not stream_reader or not image_processor:
            raise MJPEGStreamError("Server components not ready for batch search.")

        # Get one frame for the entire batch
        frame_data_dict = stream_reader.get_frame()
        if not frame_data_dict or frame_data_dict.get("frame") is None:
            raise MJPEGStreamError(
                "Failed to get a valid frame from MJPEG stream for batch search.",
                details=stream_reader.health_check(),
            )

        current_frame = frame_data_dict["frame"]
        frame_timestamp = frame_data_dict["timestamp"]

        # Save debug frame once for the batch if enabled
        if server_settings_dict.get("enable_debug_saving"):
            save_debug_frame(
                current_frame,
                server_settings_dict.get("debug_save_dir", "debug_images"),
                "batch_search_frame",  # General prefix for the batch frame
                int(server_settings_dict.get("max_debug_files", 100)),
            )

        for idx, item_request in enumerate(request_body.templates):
            item_start_time = time.time()
            item_result: Dict[str, Any] = {
                "template_path_request": item_request.template_path,
                "success": False,
            }
            try:
                # Merge global defaults from BatchSearchRequest with item-specific overrides
                # Item params take precedence over global batch params, which take precedence over server defaults (handled by ImageProcessor)

                search_params: Dict[str, Any] = {}
                search_params["filter_type"] = (
                    item_request.filter_type
                    if item_request.filter_type is not None
                    else request_body.filter_type
                )
                search_params["match_method"] = (
                    item_request.match_method
                    if item_request.match_method is not None
                    else request_body.match_method
                )
                search_params["threshold"] = (
                    item_request.threshold
                    if item_request.threshold is not None
                    else request_body.threshold
                )

                if item_request.search_region:
                    search_params.update(
                        item_request.search_region
                    )  # e.g. {"x1": ..., "y1": ...}

                # Offsets: item > default (0)
                search_params["offsetx"] = (
                    item_request.offset.get("x", 0) if item_request.offset else 0
                )
                search_params["offsety"] = (
                    item_request.offset.get("y", 0) if item_request.offset else 0
                )

                if item_request.canny_params:
                    search_params["canny_t1"] = item_request.canny_params.get("t1")
                    search_params["canny_t2"] = item_request.canny_params.get("t2")

                # Remove Nones so ImageProcessor uses its defaults (from server.ini or constants)
                final_search_params = {
                    k: v for k, v in search_params.items() if v is not None
                }

                item_match_data = image_processor.match_template(
                    current_frame, item_request.template_path, final_search_params
                )

                # Add batch-specific timing and frame info
                item_match_data.setdefault("frame_info", {})
                item_match_data["frame_info"][
                    "timestamp_epoch"
                ] = frame_timestamp  # Same frame for all items
                item_match_data["frame_info"]["item_processing_time_ms"] = (
                    time.time() - item_start_time
                ) * 1000
                item_match_data["template_path_request"] = (
                    item_request.template_path
                )  # Ensure requested path is in result

                results.append(item_match_data)

            except ImageSearchError as e:
                logger.warning(
                    f"Error processing template '{item_request.template_path}' in batch (idx {idx}): {e.error_type} - {e}"
                )
                item_result.update(
                    {"error": str(e), "error_type": e.error_type, "details": e.details}
                )
                results.append(item_result)
            except Exception as e_item:  # Catch-all for unexpected errors per item
                logger.error(
                    f"Unexpected error processing template '{item_request.template_path}' in batch (idx {idx}): {e_item}",
                    exc_info=True,
                )
                item_result.update(
                    {"error": str(e_item), "error_type": type(e_item).__name__}
                )
                results.append(item_result)

        if response:  # Set global headers for the batch response
            response.headers["X-Frame-Timestamp"] = f"{frame_timestamp:.3f}"
        return results

    # Internal Management Endpoints
    @app.put(
        "/internal/update_mjpeg_url", summary="Update the MJPEG stream URL dynamically."
    )
    async def update_mjpeg_url_endpoint(
        request_data: MJPEGURLUpdateRequest,
    ):  # Validation by Pydantic model
        stream_reader: Optional[MJPEGStreamReader] = app.state.stream_reader
        if not stream_reader:
            logger.error("Stream reader not initialized, cannot update MJPEG URL.")
            raise HTTPException(  # Use HTTPException for API error responses
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,  # Or 500 if it's an internal setup issue
                detail="Stream reader component is not available.",
            )

        logger.info(
            f"API: Received request to update MJPEG URL to: {request_data.new_url}"
        )
        try:
            # MJPEGStreamReader.update_mjpeg_url is blocking and handles rollback.
            # It raises MJPEGStreamError on failure (e.g. new URL invalid, start fails, revert fails).
            stream_reader.update_mjpeg_url(
                request_data.new_url
            )  # This can raise MJPEGStreamError

            logger.info(
                f"API: Successfully updated MJPEG URL to {request_data.new_url} and stream is active."
            )
            return {
                "status": "ok",
                "message": f"MJPEG URL updated to {request_data.new_url} and stream is active.",
            }

        except MJPEGStreamError as e:  # Catch errors from update_mjpeg_url
            logger.error(
                f"API: Failed to update MJPEG URL to '{request_data.new_url}'. Error: {e}"
            )
            # Distinguish between "new URL failed but reverted" vs "total failure"
            if "Stream reverted to" in str(e) and "and is active" in str(e):
                # New URL failed, but reverted successfully. Client's request (new URL) was not fulfilled.
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
                )
            else:
                # New URL failed, AND revert also failed OR another critical stream error. Stream is likely down.
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
                )
        except Exception as e_update:  # Catch any other unexpected error
            logger.exception(
                f"API: Unexpected error during MJPEG URL update to {request_data.new_url}:"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected internal error during URL update: {str(e_update)}",
            )

    @app.get(
        "/internal/status",
        summary="Get detailed internal status of the server instance.",
    )
    async def internal_status_endpoint():
        stream_reader: Optional[MJPEGStreamReader] = app.state.stream_reader
        app_cfg: Optional[AppConfig] = app.state.app_config
        image_proc_ok = app.state.image_processor is not None

        current_mjpeg_url = "N/A"
        if stream_reader:
            current_mjpeg_url = stream_reader.mjpeg_url
        elif app_cfg:
            current_mjpeg_url = app_cfg.mjpeg_url

        status_report = {
            "server_version": SERVER_VERSION,
            "app_config_port": app_cfg.port if app_cfg else "N/A",
            "image_processor_initialized": image_proc_ok,
            "mjpeg_url_configured": current_mjpeg_url,  # The URL the stream reader is trying to use
        }

        if not stream_reader:
            status_report["status"] = "error"
            status_report["message"] = "Stream reader component not initialized."
            status_report["mjpeg_stream_details"] = {
                "is_active": False,
                "stream_status_message": "Not Initialized",
            }
        else:
            status_report["status"] = (
                "ok"  # Base status, can be overridden by stream health
            )
            status_report["mjpeg_stream_details"] = stream_reader.health_check()
            if not status_report["mjpeg_stream_details"].get("is_active"):
                status_report["status"] = "warning"  # Or "error" depending on severity
                status_report["message"] = (
                    "MJPEG stream is not active or experiencing issues."
                )
            elif "Error" in status_report["mjpeg_stream_details"].get(
                "stream_status_message", ""
            ) or "Failed" in status_report["mjpeg_stream_details"].get(
                "stream_status_message", ""
            ):
                status_report["status"] = "error"
                status_report["message"] = (
                    f"MJPEG stream error: {status_report['mjpeg_stream_details'].get('stream_status_message')}"
                )

        if (
            not image_proc_ok and status_report["status"] != "error"
        ):  # If not already error
            status_report["status"] = "error"
            status_report["message"] = (
                status_report.get("message", "") + " Image processor not initialized."
            ).strip()

        if status_report.get("message") is None and status_report["status"] == "ok":
            status_report["message"] = "Server components operational."

        return status_report

    @app.post(
        "/internal/shutdown",
        summary="Initiate graceful shutdown of the server instance's stream.",
    )
    async def shutdown_server_stream_endpoint():  # Renamed to be more specific
        logger.info(
            "API: Received /internal/shutdown request. Initiating graceful stop of MJPEG stream."
        )
        stream_reader: Optional[MJPEGStreamReader] = app.state.stream_reader

        if stream_reader:
            stream_reader.stop()  # This is a blocking call.
            logger.info("API: MJPEG stream reader stop process completed.")
            msg = "MJPEG stream shutdown initiated successfully."
        else:
            logger.info(
                "API: MJPEG stream reader was not initialized or already None during shutdown request."
            )
            msg = (
                "MJPEG stream reader not active or not initialized; no stream to stop."
            )

        # This endpoint only stops the stream. Uvicorn/process shutdown is external.
        return {"status": "ok", "message": msg}

    # Lifecycle events
    @app.on_event("startup")
    async def startup_event():
        logger.info(
            f"FastAPI application (Version {SERVER_VERSION}, Port {app.state.app_config.port if app.state.app_config else 'N/A'}) starting up..."
        )
        # Stream is started during create_app's MJPEGStreamReader initialization.
        # ImageProcessor is also initialized in create_app.
        # Any further app-level startup logic can go here.

    @app.on_event("shutdown")
    async def shutdown_event_handler():
        port_info = app.state.app_config.port if app.state.app_config else "N/A"
        logger.info(
            f"FastAPI application (Version {SERVER_VERSION}, Port {port_info}) shutting down..."
        )
        stream_reader: Optional[MJPEGStreamReader] = app.state.stream_reader
        if stream_reader:
            logger.info(
                f"Stopping MJPEG stream reader for {stream_reader.mjpeg_url} during app shutdown..."
            )
            stream_reader.stop()
            logger.info("MJPEG stream reader stopped.")

        if app.state.image_processor:
            logger.info("Cleaning up image processor (if applicable)...")
            # If ImageProcessor had explicit cleanup, e.g., app.state.image_processor.cleanup()
            app.state.image_processor = None  # Allow GC

        logger.info(f"Application shutdown process completed for port {port_info}.")

    logger.info(f"FastAPI application (Version {SERVER_VERSION}) definition completed.")
    return app
