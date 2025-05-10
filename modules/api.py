# -*- coding: utf-8 -*-
"""
API Module
Defines FastAPI endpoints and request handlers for the image matching service
"""
import glob
import logging
import os
import time
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Query, Response, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from modules.constants import SERVER_VERSION
from modules.image_processor import ImageProcessor
from modules.mjpeg import MJPEGStreamReader
from modules.state_managers import FrameDataCache
from modules.utils import (
    parse_ini_file,
    parse_query_params,
    get_match_range_params,
    validate_filter_type,
    validate_match_method,
    save_debug_frame,
    clean_debug_images
)

logger = logging.getLogger(__name__)

# 移除全局变量
# stream_reader = None
# image_processor = None
# server_config = {}

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    mjpeg_active_target: bool
    frame_valid: bool
    frame_age_seconds: float
    frame_resolution: str
    server_version: str

class MJPEGURLUpdateRequest(BaseModel):
    """Request model for updating MJPEG URL"""
    new_url: str

def create_app(config) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        config: AppConfig instance or command line arguments
        
    Returns:
        Configured FastAPI application
    """
    # 移除全局变量的引用
    # global stream_reader, image_processor, server_config
    
    # Create FastAPI app
    app = FastAPI(
        title="Image Matching API",
        description="API for template-based image matching in MJPEG streams",
        version=SERVER_VERSION
    )
    
    # 转换配置对象为字典，适配两种不同的配置输入方式
    if hasattr(config, 'server_config'):
        # 如果是 AppConfig 实例
        server_config = config.server_config
        mjpeg_url = config.mjpeg_url
    else:
        # 如果是命令行参数对象
        server_config = vars(config) if hasattr(config, '__dict__') else dict(config)
        mjpeg_url = server_config['mjpeg_url']
    
    # 创建帧数据缓存
    frame_cache = FrameDataCache()
    
    # 初始化 MJPEGStreamReader 并存储在 app.state 中
    app.state.stream_reader = MJPEGStreamReader(
        mjpeg_url=mjpeg_url,
        idle_timeout=server_config.get('idle_timeout', 300.0),
        wakeup_timeout=server_config.get('wakeup_timeout', 15.0)
    )
    
    # 初始化 ImageProcessor 并存储在 app.state 中
    app.state.image_processor = ImageProcessor(server_config)
    
    # 存储 server_config 在 app.state 中
    app.state.server_config = server_config
    
    # 如果使用 AppConfig 实例，更新引用
    if hasattr(config, 'stream_reader') and hasattr(config, 'image_processor'):
        config.stream_reader = app.state.stream_reader
        config.image_processor = app.state.image_processor
    
    # Create debug directory if enabled
    if server_config.get('enable_debug_saving', False) and server_config.get('debug_save_dir'):
        os.makedirs(server_config['debug_save_dir'], exist_ok=True)
        logger.info(f"Debug image saving enabled to directory: {server_config['debug_save_dir']}")
    
    # Define API routes
    @app.get("/health")
    async def health_check() -> HealthResponse:
        """
        Health check endpoint
        
        Returns:
            HealthResponse: Health status information
        """
        stream_reader = app.state.stream_reader
        
        if not stream_reader:
            return HealthResponse(
                status="error",
                message="MJPEG stream reader not initialized",
                mjpeg_active_target=False,
                frame_valid=False,
                frame_age_seconds=9999999.0,
                frame_resolution="unknown",
                server_version=SERVER_VERSION
            )
        
        # Get stream health data
        health_data = stream_reader.health_check()
        
        # Determine overall status
        status_val = "ok"
        message = f"Running (Frame @ {time.strftime('%H:%M:%S')})"
        
        # Check frame age
        if health_data['frame_age_seconds'] > server_config.get('wakeup_timeout', 15.0):
            status_val = "warning"
            message = f"Frame age ({health_data['frame_age_seconds']:.1f}s) exceeds timeout"
        
        # Check if frame is valid
        if not health_data['frame_valid']:
            status_val = "error"
            message = "No valid frame available"
        
        return HealthResponse(
            status=status_val,
            message=message,
            mjpeg_active_target=health_data['active'],
            frame_valid=health_data['frame_valid'],
            frame_age_seconds=health_data['frame_age_seconds'],
            frame_resolution=health_data['frame_resolution'],
            server_version=SERVER_VERSION
        )
    
    @app.get("/search_ini")
    async def search_ini(
        ini_path: str = Query(..., description="Absolute path to INI file"),
        response: Response = None
    ) -> Dict[str, Any]:
        """
        Search for template defined in INI file
        
        Args:
            ini_path: Absolute path to INI configuration file
            response: FastAPI Response object
            
        Returns:
            Match result dictionary
        """
        start_time = time.time()
        
        # 从 app.state 获取实例
        stream_reader = app.state.stream_reader
        image_processor = app.state.image_processor
        server_config = app.state.server_config
        
        # Ensure stream_reader is initialized
        if not stream_reader:
            raise HTTPException(
                status_code=503,
                detail="MJPEG stream reader not initialized"
            )
            
        try:
            # Parse INI file
            logger.debug(f"Parsing INI file: {ini_path}")
            ini_params = parse_ini_file(ini_path)
            
            # Get template path from INI parameters
            template_path = ini_params.get('template_path')
            if not template_path or not os.path.exists(template_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Template file not found: {template_path}"
                )
                
            # Prepare search parameters
            search_params = {}
            
            # Copy and normalize INI parameters
            for key, value in ini_params.items():
                if key in ('template_path', 'ini_path'):
                    continue  # Skip these special keys
                
                # Add parameter to search parameters
                search_params[key] = value
                
            # Get current frame from stream
            frame, frame_timestamp, frame_dimensions = stream_reader.get_frame()
            
            # Check if we got a valid frame
            if frame is None:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to get frame from MJPEG stream"
                )
                
            # Save debug frame if enabled
            if server_config.get('enable_debug_saving', False) and server_config.get('debug_save_dir'):
                save_debug_frame(
                    frame,
                    server_config['debug_save_dir'],
                    "search_ini_request",
                    server_config.get('max_debug_files', 100)
                )
                
            # Perform template matching
            result = image_processor.match_template(frame, template_path, search_params)
            
            # Add processing time header
            if response:
                response.headers["X-Processing-Time"] = f"{time.time() - start_time:.3f}"
                response.headers["X-Frame-Timestamp"] = f"{frame_timestamp:.3f}"
                
            return result
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=str(e)
            )
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            logger.exception(f"Error in search_ini: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    @app.get("/search")
    @app.get("/")
    async def search(
        request: Request,
        img: str = Query(..., description="Absolute path to template image"),
        filter_type: Optional[str] = Query(None, description="Filter type (none/canny)"),
        match_method: Optional[str] = Query(None, description="Match method"),
        threshold: Optional[float] = Query(None, description="Match threshold (0.0-1.0)"),
        x1: Optional[int] = Query(None, description="Search region X1 coordinate"),
        y1: Optional[int] = Query(None, description="Search region Y1 coordinate"),
        x2: Optional[int] = Query(None, description="Search region X2 coordinate"),
        y2: Optional[int] = Query(None, description="Search region Y2 coordinate"),
        offsetx: Optional[int] = Query(None, description="Offset X to add to result coordinates"),
        offsety: Optional[int] = Query(None, description="Offset Y to add to result coordinates"),
        waitForRecheck: Optional[float] = Query(0.0, description="Wait time for recheck"),
        canny_t1: Optional[int] = Query(None, description="Canny threshold 1"),
        canny_t2: Optional[int] = Query(None, description="Canny threshold 2"),
        response: Response = None
    ) -> Dict[str, Any]:
        """
        Search for template in current frame
        
        Args:
            request: FastAPI Request object
            img: Absolute path to template image (can be glob pattern)
            filter_type: Filter type to apply
            match_method: Match method to use
            threshold: Match threshold
            x1, y1, x2, y2: Search region coordinates
            offsetx, offsety: Offset to add to result coordinates
            waitForRecheck: Wait time for recheck verification
            canny_t1, canny_t2: Canny filter thresholds
            response: FastAPI Response object
            
        Returns:
            Match result dictionary
        """
        start_time = time.time()
        
        # Ensure stream_reader is initialized
        if not app.state.stream_reader:
            raise HTTPException(
                status_code=503,
                detail="MJPEG stream reader not initialized"
            )
            
        try:
            # Handle glob pattern in img parameter
            template_paths = []
            if '*' in img or '?' in img:
                template_paths = glob.glob(img)
                if not template_paths:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No files match pattern: {img}"
                    )
            else:
                if not os.path.exists(img):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Template file not found: {img}"
                    )
                template_paths = [img]
                
            # Prepare search parameters from query parameters
            # Get all query parameters
            query_params = dict(request.query_params)
            
            # Parse and convert query parameters
            search_params = parse_query_params(query_params)
            
            # Validate and set filter_type
            if filter_type:
                search_params['filter_type'] = validate_filter_type(filter_type)
            else:
                search_params['filter_type'] = server_config.get('default_filter_type', 'none')
                
            # Validate and set match_method
            if match_method:
                search_params['match_method'] = validate_match_method(match_method)
            else:
                search_params['match_method'] = server_config.get('default_match_method', 'ccoeff_normed')
                
            # Set threshold
            if threshold is not None:
                search_params['threshold'] = max(0.0, min(1.0, threshold))
            else:
                search_params['threshold'] = server_config.get('default_threshold', 0.8)
                
            # Set Canny thresholds if filter_type is 'canny'
            if search_params['filter_type'] == 'canny':
                if canny_t1 is not None:
                    search_params['canny_t1'] = canny_t1
                else:
                    search_params['canny_t1'] = server_config.get('default_canny_t1', 50)
                    
                if canny_t2 is not None:
                    search_params['canny_t2'] = canny_t2
                else:
                    search_params['canny_t2'] = server_config.get('default_canny_t2', 150)
                    
            # Add search region parameters
            if x1 is not None:
                search_params['match_range_x1'] = x1
            if y1 is not None:
                search_params['match_range_y1'] = y1
            if x2 is not None:
                search_params['match_range_x2'] = x2
            if y2 is not None:
                search_params['match_range_y2'] = y2
                
            # Add offset parameters
            if offsetx is not None:
                search_params['offset_x'] = offsetx
            if offsety is not None:
                search_params['offset_y'] = offsety
                
            # Add waitForRecheck
            if waitForRecheck is not None and waitForRecheck > 0:
                search_params['waitforrecheck'] = waitForRecheck
                
            # Get current frame from stream
            frame, frame_timestamp, frame_dimensions = app.state.stream_reader.get_frame()
            
            # Check if we got a valid frame
            if frame is None:
                raise HTTPException(
                    status_code=503,
                    detail="Failed to get frame from MJPEG stream"
                )
                
            # Save debug frame if enabled
            if server_config.get('enable_debug_saving', False) and server_config.get('debug_save_dir'):
                save_debug_frame(
                    frame,
                    server_config['debug_save_dir'],
                    "search_request",
                    server_config.get('max_debug_files', 100)
                )
                
            # Try each template path until we find a match or exhaust all templates
            result = None
            for template_path in template_paths:
                try:
                    # Perform template matching
                    match_result = app.state.image_processor.match_template(frame, template_path, search_params)
                    
                    # If found or this is the last template, use this result
                    if match_result.get('found', False) or template_path == template_paths[-1]:
                        result = match_result
                        break
                        
                except Exception as e:
                    logger.warning(f"Error matching template {template_path}: {e}")
                    if template_path == template_paths[-1]:
                        # If this is the last template, re-raise the exception
                        raise
            
            # Add processing time header
            if response:
                response.headers["X-Processing-Time"] = f"{time.time() - start_time:.3f}"
                response.headers["X-Frame-Timestamp"] = f"{frame_timestamp:.3f}"
                
            return result
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise HTTPException(
                status_code=404,
                detail=str(e)
            )
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        except Exception as e:
            logger.exception(f"Error in search: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Generic exception handler for all uncaught exceptions"""
        logger.exception(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {str(exc)}"}
        )
    
    # Log successful app creation
    logger.info(f"FastAPI application created (Version {SERVER_VERSION})")
    
    # 添加内部 API 接口用于管理
    @app.put("/internal/update_mjpeg_url")
    async def update_mjpeg_url(request: MJPEGURLUpdateRequest):
        """
        Internal endpoint to update the MJPEG URL.
        
        Args:
            request: Request containing the new MJPEG URL
            
        Returns:
            Status of the update operation
        """
        stream_reader = app.state.stream_reader
        if not stream_reader:
            raise HTTPException(
                status_code=500,
                detail="Stream reader not initialized"
            )
        
        logger.info(f"Received request to update MJPEG URL to: {request.new_url}")
        try:
            old_url = stream_reader.mjpeg_url
            # 更新 MJPEG URL
            stream_reader.mjpeg_url = request.new_url
            # 停止当前流并使用新 URL 重新开始
            stream_reader.stop()
            success = stream_reader.start()
            
            if success:
                logger.info(f"Successfully updated MJPEG URL from {old_url} to {request.new_url}")
                return {"status": "ok", "message": f"MJPEG URL updated to {request.new_url}"}
            else:
                logger.error(f"Failed to start stream with new URL: {request.new_url}")
                # 尝试回退到旧 URL
                stream_reader.mjpeg_url = old_url
                stream_reader.start()
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to start stream with new URL: {request.new_url}"
                )
        except Exception as e:
            logger.exception(f"Error updating MJPEG URL to {request.new_url}:")
            raise HTTPException(
                status_code=500,
                detail=f"Internal error updating URL: {str(e)}"
            )
    
    @app.get("/internal/status")
    async def internal_status():
        """
        Internal endpoint to get detailed instance status.
        
        Returns:
            Detailed status information about the instance
        """
        stream_reader = app.state.stream_reader
        if not stream_reader:
            return {
                "status": "error",
                "message": "Stream reader not initialized"
            }
        
        # 获取健康检查数据
        health_data = stream_reader.health_check()
        
        # 添加更多详细信息
        return {
            "status": "ok",
            "health_data": health_data,
            "mjpeg_url": stream_reader.mjpeg_url,
            "server_version": SERVER_VERSION
        }
    
    # 注册应用关闭事件处理器
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on application shutdown"""
        logger.info("Application shutdown event triggered")
        
        # 关闭视频流
        if app.state.stream_reader:
            logger.info("Stopping stream reader...")
            app.state.stream_reader.stop()
            app.state.stream_reader = None
            logger.info("Stream reader stopped")
            
        # 清理图像处理器
        if app.state.image_processor:
            logger.info("Cleaning up image processor...")
            # 如果 ImageProcessor 有需要清理的资源，在这里处理
            app.state.image_processor = None
            logger.info("Image processor cleaned up")
    
    return app 