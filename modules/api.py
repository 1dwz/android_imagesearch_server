# -*- coding: utf-8 -*-
"""
API Module
Defines FastAPI endpoints and request handlers for the image matching service
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from modules.constants import SERVER_VERSION
from modules.datatypes import (
    ImageSearchError,
    ParameterValidator,
    ValidationError,
)
from modules.image_processor import ImageProcessor
from modules.mjpeg import MJPEGStreamReader
from modules.state_managers import FrameDataCache
from modules.utils import (
    parse_ini_file,
    save_debug_frame,
)

logger = logging.getLogger(__name__)

# 移除全局变量
# stream_reader = None
# image_processor = None
# server_config = {}


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field(..., description="服务器状态 (ok/warning/error)")
    message: str = Field(..., description="状态消息")
    mjpeg_active_target: bool = Field(..., description="MJPEG 流是否活跃")
    frame_valid: bool = Field(..., description="当前帧是否有效")
    frame_age_seconds: float = Field(..., description="当前帧的年龄（秒）")
    frame_resolution: str = Field(..., description="帧分辨率")
    server_version: str = Field(..., description="服务器版本")

    @validator("status")
    def validate_status(cls, v):
        """验证状态值"""
        if v not in ["ok", "warning", "error"]:
            raise ValueError("状态必须是 ok、warning 或 error 之一")
        return v


class MJPEGURLUpdateRequest(BaseModel):
    """MJPEG URL 更新请求模型"""

    new_url: str = Field(..., description="新的 MJPEG URL")

    @validator("new_url")
    def validate_url(cls, v):
        """验证 URL 格式"""
        if not v.startswith(("http://", "https://", "rtsp://")):
            raise ValueError("URL 必须以 http://, https:// 或 rtsp:// 开头")
        return v


class BatchSearchRequest(BaseModel):
    """批量搜索请求模型"""

    templates: List[str] = Field(..., description="模板图像路径列表")
    filter_type: Optional[str] = Field(None, description="过滤器类型 (none/canny)")
    match_method: Optional[str] = Field(None, description="匹配方法")
    threshold: Optional[float] = Field(None, description="匹配阈值 (0.0-1.0)")
    search_region: Optional[Dict[str, int]] = Field(None, description="搜索区域")
    offset: Optional[Dict[str, int]] = Field(None, description="结果坐标偏移")
    canny_params: Optional[Dict[str, int]] = Field(
        None, description="Canny 边缘检测参数"
    )

    @validator("templates")
    def validate_templates(cls, v):
        """验证模板路径列表"""
        if not v:
            raise ValueError("模板列表不能为空")
        for path in v:
            ParameterValidator.validate_image_format(path)
            if not os.path.isfile(path):
                raise ValueError(f"模板文件不存在: {path}")
        return v

    @validator("threshold")
    def validate_threshold(cls, v):
        """验证阈值"""
        if v is not None:
            return ParameterValidator.validate_threshold(v)
        return v


class ErrorResponse(BaseModel):
    """错误响应模型"""

    error: str = Field(..., description="错误消息")
    error_type: str = Field(..., description="错误类型")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")


def create_app(config) -> FastAPI:
    """
    创建并配置 FastAPI 应用

    Args:
        config: AppConfig 实例或命令行参数

    Returns:
        配置好的 FastAPI 应用
    """
    # 移除全局变量的引用
    # global stream_reader, image_processor, server_config

    # Create FastAPI app
    app = FastAPI(
        title="图像匹配 API",
        description="用于 MJPEG 流中基于模板的图像匹配的 API",
        version=SERVER_VERSION,
    )

    # 转换配置对象为字典，适配两种不同的配置输入方式
    if hasattr(config, "server_config"):
        # 如果是 AppConfig 实例
        server_config = config.server_config
        mjpeg_url = config.mjpeg_url
    else:
        # 如果是命令行参数对象
        server_config = vars(config) if hasattr(config, "__dict__") else dict(config)
        mjpeg_url = server_config["mjpeg_url"]

    # 验证 MJPEG URL
    try:
        mjpeg_url = ParameterValidator.validate_mjpeg_url(mjpeg_url)
    except ValidationError as e:
        logger.error(f"无效的 MJPEG URL: {e}")
        raise

    # 创建帧数据缓存
    FrameDataCache()

    # 初始化 MJPEGStreamReader 并存储在 app.state 中
    app.state.stream_reader = MJPEGStreamReader(
        mjpeg_url=mjpeg_url,
        idle_timeout=server_config.get("idle_timeout", 300.0),
        wakeup_timeout=server_config.get("wakeup_timeout", 15.0),
    )

    # 初始化 ImageProcessor 并存储在 app.state 中
    app.state.image_processor = ImageProcessor(server_config)

    # 存储 server_config 在 app.state 中
    app.state.server_config = server_config

    # 如果使用 AppConfig 实例，更新引用
    if hasattr(config, "stream_reader") and hasattr(config, "image_processor"):
        config.stream_reader = app.state.stream_reader
        config.image_processor = app.state.image_processor

    # Create debug directory if enabled
    if server_config.get("enable_debug_saving", False) and server_config.get(
        "debug_save_dir"
    ):
        os.makedirs(server_config["debug_save_dir"], exist_ok=True)
        logger.info(
            f"Debug image saving enabled to directory: {server_config['debug_save_dir']}"
        )

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
                server_version=SERVER_VERSION,
            )

        # Get stream health data
        health_data = stream_reader.health_check()

        # Determine overall status
        status_val = "ok"
        message = f"Running (Frame @ {time.strftime('%H:%M:%S')})"

        # Check frame age
        if health_data["frame_age_seconds"] > server_config.get("wakeup_timeout", 15.0):
            status_val = "warning"
            message = (
                f"Frame age ({health_data['frame_age_seconds']:.1f}s) exceeds timeout"
            )

        # Check if frame is valid
        if not health_data["frame_valid"]:
            status_val = "error"
            message = "No valid frame available"

        return HealthResponse(
            status=status_val,
            message=message,
            mjpeg_active_target=health_data["active"],
            frame_valid=health_data["frame_valid"],
            frame_age_seconds=health_data["frame_age_seconds"],
            frame_resolution=health_data["frame_resolution"],
            server_version=SERVER_VERSION,
        )

    @app.get("/search_ini")
    async def search_ini(
        ini_path: str = Query(..., description="Absolute path to INI file"),
        response: Response = None,
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

        # Strip whitespace from ini_path
        ini_path = ini_path.strip()

        # 从 app.state 获取实例
        stream_reader = app.state.stream_reader
        image_processor = app.state.image_processor
        server_config = app.state.server_config

        # Ensure stream_reader is initialized
        if not stream_reader:
            raise HTTPException(
                status_code=503, detail="MJPEG stream reader not initialized"
            )

        try:
            # Parse INI file
            logger.debug(f"Parsing INI file: {ini_path}")
            ini_params = parse_ini_file(ini_path)

            # Get template path from INI parameters
            template_path = ini_params.get("template_path")
            if not template_path or not os.path.exists(template_path):
                raise HTTPException(
                    status_code=404, detail=f"Template file not found: {template_path}"
                )

            # Prepare search parameters
            search_params = {}

            # Copy and normalize INI parameters
            for key, value in ini_params.items():
                if key in ("template_path", "ini_path"):
                    continue  # Skip these special keys

                # Add parameter to search parameters
                search_params[key] = value

            # Get current frame from stream
            frame_data = stream_reader.get_frame()
            if frame_data is None or frame_data.get("frame") is None:
                raise HTTPException(
                    status_code=503, detail="Failed to get frame from MJPEG stream"
                )
            frame = frame_data["frame"]
            frame_timestamp = frame_data["timestamp"]

            # Save debug frame if enabled
            if server_config.get("enable_debug_saving", False) and server_config.get(
                "debug_save_dir"
            ):
                save_debug_frame(
                    frame,
                    server_config["debug_save_dir"],
                    "search_ini_request",
                    server_config.get("max_debug_files", 100),
                )

            # Perform template matching
            result = image_processor.match_template(frame, template_path, search_params)

            # Add processing time header
            if response:
                response.headers["X-Processing-Time"] = (
                    f"{time.time() - start_time:.3f}"
                )
                response.headers["X-Frame-Timestamp"] = f"{frame_timestamp:.3f}"

            return result

        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in search_ini: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )

    async def _execute_single_search(
        template_path: str,
        params: Dict[str, Any],
        stream_reader: MJPEGStreamReader,
        image_processor: ImageProcessor,
        server_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行单个模板的搜索操作。

        Args:
            template_path: 模板图像路径
            params: 搜索参数字典
            stream_reader: MJPEG 流读取器实例
            image_processor: 图像处理器实例
            server_config: 服务器配置字典

        Returns:
            搜索结果字典

        Raises:
            ValidationError: 当参数验证失败时
            ResourceNotFoundError: 当模板文件不存在时
            MJPEGStreamError: 当流操作失败时
            ImageSearchError: 当图像搜索相关操作失败时
        """
        # 获取帧
        frame_data = stream_reader.get_frame()
        if frame_data is None or frame_data.get("frame") is None:
            raise MJPEGStreamError("无法从 MJPEG 流获取有效帧")

        current_frame = frame_data["frame"]
        frame_timestamp = frame_data["timestamp"]

        # 调试帧保存
        if server_config.get("enable_debug_saving") and server_config.get(
            "debug_save_dir"
        ):
            save_debug_frame(
                current_frame,
                server_config["debug_save_dir"],
                "search_request",
                server_config.get("max_debug_files", 100),
            )

        # 执行模板匹配
        result = image_processor.match_template(current_frame, template_path, params)

        # 添加时间戳到结果中
        if isinstance(result, dict):
            result.setdefault("processing_details", {})
            result["processing_details"]["frame_timestamp_epoch"] = frame_timestamp

        return result

    @app.get("/search")
    @app.get("/")
    async def search(
        request: Request,
        img: str = Query(..., description="模板图像的绝对路径"),
        filter_type: Optional[str] = Query(None, description="过滤器类型 (none/canny)"),
        match_method: Optional[str] = Query(None, description="匹配方法"),
        threshold: Optional[float] = Query(None, description="匹配阈值 (0.0-1.0)"),
        x1: Optional[int] = Query(None, description="搜索区域 X1 坐标"),
        y1: Optional[int] = Query(None, description="搜索区域 Y1 坐标"),
        x2: Optional[int] = Query(None, description="搜索区域 X2 坐标"),
        y2: Optional[int] = Query(None, description="搜索区域 Y2 坐标"),
        offsetx: Optional[int] = Query(None, description="添加到结果坐标的偏移 X"),
        offsety: Optional[int] = Query(None, description="添加到结果坐标的偏移 Y"),
        canny_t1: Optional[int] = Query(None, description="Canny 阈值 1"),
        canny_t2: Optional[int] = Query(None, description="Canny 阈值 2"),
        response: Response = None,
    ) -> Dict[str, Any]:
        """
        在当前帧中搜索模板图像。
        """
        start_time = time.time()
        try:
            # 构建搜索参数
            search_params = {
                "filter_type": filter_type,
                "match_method": match_method,
                "threshold": threshold,
            }

            # 添加搜索区域参数（只有当所有参数都提供或都未提供时才添加）
            if all(coord is not None for coord in [x1, y1, x2, y2]):
                search_params.update(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
            elif any(coord is not None for coord in [x1, y1, x2, y2]):
                # 如果只提供了部分参数，这将在 validate_search_region 中被捕获并抛出错误
                search_params.update(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

            # 添加偏移参数（使用默认值 0）
            search_params.update(
                {
                    "offsetx": offsetx if offsetx is not None else 0,
                    "offsety": offsety if offsety is not None else 0,
                }
            )

            # 添加 Canny 参数（只有当两个参数都提供或都未提供时才添加）
            if canny_t1 is not None and canny_t2 is not None:
                search_params.update(
                    {
                        "canny_t1": canny_t1,
                        "canny_t2": canny_t2,
                    }
                )
            elif canny_t1 is not None or canny_t2 is not None:
                # 如果只提供了一个参数，这将在 validate_canny_params 中被捕获并抛出错误
                search_params.update(
                    {
                        "canny_t1": canny_t1,
                        "canny_t2": canny_t2,
                    }
                )

            # 执行搜索
            result = await _execute_single_search(
                img,
                search_params,
                app.state.stream_reader,
                app.state.image_processor,
                app.state.server_config,
            )

            # 设置响应头
            if response:
                response.headers["X-Processing-Time"] = (
                    f"{time.time() - start_time:.3f}"
                )
                if result and isinstance(result, dict):
                    frame_timestamp = result.get("processing_details", {}).get(
                        "frame_timestamp_epoch"
                    )
                    if frame_timestamp:
                        response.headers["X-Frame-Timestamp"] = f"{frame_timestamp:.3f}"

            return result

        except Exception as e:
            if isinstance(
                e,
                (
                    ValidationError,
                    ResourceNotFoundError,
                    MJPEGStreamError,
                    ImageSearchError,
                ),
            ):
                raise
            logger.exception(f"搜索操作时发生未知错误: {e}")
            raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

    @app.exception_handler(ImageSearchError)
    async def image_search_exception_handler(request: Request, exc: ImageSearchError):
        """处理图像搜索相关错误"""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=str(exc), error_type=exc.error_type, details=exc.details
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """处理通用错误"""
        logger.exception("未处理的异常")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="服务器内部错误",
                error_type="InternalError",
                details={"exception": str(exc)},
            ).dict(),
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
            raise HTTPException(status_code=500, detail="Stream reader not initialized")

        logger.info(f"Received request to update MJPEG URL to: {request.new_url}")
        try:
            old_url = stream_reader.mjpeg_url
            # 更新 MJPEG URL
            stream_reader.mjpeg_url = request.new_url
            # 停止当前流并使用新 URL 重新开始
            stream_reader.stop()
            success = stream_reader.start()

            if success:
                logger.info(
                    f"Successfully updated MJPEG URL from {old_url} to {request.new_url}"
                )
                return {
                    "status": "ok",
                    "message": f"MJPEG URL updated to {request.new_url}",
                }
            else:
                logger.error(f"Failed to start stream with new URL: {request.new_url}")
                # 尝试回退到旧 URL
                stream_reader.mjpeg_url = old_url
                stream_reader.start()
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to start stream with new URL: {request.new_url}",
                )
        except Exception as e:
            logger.exception(f"Error updating MJPEG URL to {request.new_url}:")
            raise HTTPException(
                status_code=500, detail=f"Internal error updating URL: {str(e)}"
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
            return {"status": "error", "message": "Stream reader not initialized"}

        # 获取健康检查数据
        health_data = stream_reader.health_check()

        # 添加更多详细信息
        return {
            "status": "ok",
            "health_data": health_data,
            "mjpeg_url": stream_reader.mjpeg_url,
            "server_version": SERVER_VERSION,
        }

    @app.post("/internal/shutdown")
    async def shutdown_server():
        """
        接收关闭信号并执行清理操作。

        此端点用于在外部触发服务器的应用级别清理（如停止 MJPEG 流）。
        实际进程终止由 manager.py 负责。
        """
        logger.info(f"接收到关闭信号，正在停止 MJPEG 流读取器...")
        if app.state.stream_reader:
            app.state.stream_reader.stop()
            logger.info("MJPEG 流读取器已停止。")
        # 返回一个消息，表明清理已启动，manager 将负责进程终止
        return {
            "message": "Shutdown process initiated, MJPEG stream reader stopping.",
            "status": "ok",
        }

    @app.post("/batch_search")
    async def batch_search(request: BatchSearchRequest) -> List[Dict[str, Any]]:
        """
        批量搜索多个模板。

        Args:
            request: BatchSearchRequest 模型实例

        Returns:
            List[Dict[str, Any]]: 每个模板的搜索结果列表
        """
        results = []
        for template_path in request.templates:
            try:
                # 构建搜索参数
                search_params = {
                    "filter_type": request.filter_type,
                    "match_method": request.match_method,
                    "threshold": request.threshold,
                }

                # 添加搜索区域参数
                if request.search_region:
                    search_params.update(request.search_region)

                # 添加偏移参数
                if request.offset:
                    search_params.update(
                        {
                            "offsetx": request.offset.get("x", 0),
                            "offsety": request.offset.get("y", 0),
                        }
                    )

                # 添加 Canny 参数
                if request.canny_params:
                    search_params.update(
                        {
                            "canny_t1": request.canny_params.get("t1"),
                            "canny_t2": request.canny_params.get("t2"),
                        }
                    )

                # 执行搜索
                result = await _execute_single_search(
                    template_path,
                    search_params,
                    app.state.stream_reader,
                    app.state.image_processor,
                    app.state.server_config,
                )
                results.append(result)

            except Exception as e:
                # 记录错误但继续处理其他模板
                logger.error(f"处理模板 {template_path} 时出错: {e}")
                error_details = {
                    "template_path": template_path,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                if isinstance(e, ImageSearchError):
                    error_details["details"] = e.details
                results.append(error_details)

        return results

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
