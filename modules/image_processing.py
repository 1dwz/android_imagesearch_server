# -*- coding: utf-8 -*-
"""
Image processing module with enhanced resource management and error handling.
Provides template matching and image filtering functionality for the search server.
"""
import asyncio
import functools
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from modules.datatypes import FilterProcessingError, FilterType, MatchMethod
from modules.utils import _load_and_process_template

logger = logging.getLogger(__name__)

DEFAULT_FILTER_TYPE = FilterType.NONE
DEFAULT_MATCH_METHOD_ENUM = MatchMethod.CCoeffNormed

# 结果缓存 - 优化性能
_result_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
_RESULT_CACHE_TTL = 1.0  # 缓存结果的有效时间（秒），从0.5秒增加到1.0秒以提高缓存命中率
_MAX_CACHE_SIZE = 50  # 最大缓存结果数量，从20增加到50以减少缓存未命中情况


def _generate_cache_key(template_path: str, params: Dict[str, Any]) -> str:
    """生成缓存键，用于结果缓存"""
    # 基础键：模板路径
    key_parts = [str(template_path)]

    # 添加主要搜索参数
    filter_type = str(params.get("filter_type", "none"))
    key_parts.append(f"filter={filter_type}")

    match_method = str(params.get("match_method", "ccoeff_normed"))
    key_parts.append(f"method={match_method}")

    threshold = str(params.get("threshold", "0.8"))
    key_parts.append(f"thresh={threshold}")

    # 添加搜索区域（如果有）
    x1 = params.get("match_range_x1")
    y1 = params.get("match_range_y1")
    x2 = params.get("match_range_x2")
    y2 = params.get("match_range_y2")

    if all(v is not None for v in [x1, y1, x2, y2]):
        key_parts.append(f"region={x1},{y1},{x2},{y2}")

    return "|".join(key_parts)


def _clear_expired_cache_entries():
    """清除过期的缓存结果"""
    global _result_cache

    current_time = time.time()
    expired_keys = [
        key
        for key, (_, timestamp) in _result_cache.items()
        if current_time - timestamp > _RESULT_CACHE_TTL
    ]

    for key in expired_keys:
        _result_cache.pop(key, None)

    # 如果缓存太大，删除最旧的条目
    if len(_result_cache) > _MAX_CACHE_SIZE:
        sorted_keys = sorted(_result_cache.keys(), key=lambda k: _result_cache[k][1])
        oldest_keys = sorted_keys[: len(_result_cache) - _MAX_CACHE_SIZE]
        for key in oldest_keys:
            _result_cache.pop(key, None)


def process_template_match(
    template: np.ndarray,
    frame: np.ndarray,
    params: Dict[str, Any],
    server_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process template matching with robust error handling. This function performs the actual
    synchronous OpenCV operations and is intended to be run in an executor.

    Args:
        template: Template image
        frame: Source frame (grayscale)
        params: Matching parameters, including search range, offset, etc.
        server_config: Server configuration dictionary (converted from Namespace)

    Returns:
        Match results dictionary
    """
    # 减少不必要的日志，只在调试时记录
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Processing template match. Template shape: {template.shape}, Frame shape: {frame.shape}"
        )
    try:
        # Get processing parameters
        filter_type_str = params.get(
            "filter_type", server_config.get("default_filter_type", "none")
        )
        match_method_str = params.get(
            "match_method", server_config.get("default_match_method", "ccoeff_normed")
        )
        threshold = params.get("threshold", server_config.get("default_threshold", 0.8))

        # 快速转换，减少重复验证
        filter_type = FilterType(filter_type_str.lower())
        match_method = MatchMethod(match_method_str.lower())

        # Extract search range parameters
        x1 = params.get("match_range_x1")
        y1 = params.get("match_range_y1")
        x2 = params.get("match_range_x2")
        y2 = params.get("match_range_y2")

        # Apply search range by cropping the frame
        search_frame = frame
        frame_height, frame_width = frame.shape[:2]

        # Initialize final coordinates to full frame initially
        final_x1, final_y1, final_x2, final_y2 = 0, 0, frame_width, frame_height

        # Flag to indicate if a search range was provided in parameters
        range_provided = all(v is not None for v in [x1, y1, x2, y2])
        full_search = True  # Assume full search initially

        # Validate and apply search range if provided
        if range_provided:
            # 只在调试时记录详细信息
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Search range provided: ({x1},{y1})-({x2},{y2}). Validating and applying."
                )
            # Ensure coordinates are within frame bounds and form a valid rectangle
            valid_x1 = max(0, min(int(x1), frame_width - 1)) if x1 is not None else 0
            valid_y1 = max(0, min(int(y1), frame_height - 1)) if y1 is not None else 0
            valid_x2 = (
                max(0, min(int(x2), frame_width - 1)) if x2 is not None else frame_width
            )
            valid_y2 = (
                max(0, min(int(y2), frame_height - 1))
                if y2 is not None
                else frame_height
            )

            # Ensure x1 < x2 and y1 < y2
            potential_final_x1 = min(valid_x1, valid_x2)
            potential_final_y1 = min(valid_y1, valid_y2)
            potential_final_x2 = max(valid_x1, valid_x2)
            potential_final_y2 = max(valid_y1, valid_y2)

            # Check if the resulting region is at least 1x1 pixel
            if (
                potential_final_x2 > potential_final_x1
                and potential_final_y2 > potential_final_y1
            ):
                final_x1, final_y1, final_x2, final_y2 = (
                    potential_final_x1,
                    potential_final_y1,
                    potential_final_x2,
                    potential_final_y2,
                )
                search_frame = frame[final_y1:final_y2, final_x1:final_x2]
                full_search = False  # Not a full search if range was applied
                # 只在调试时记录详细信息
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Validated search range: ({final_x1},{final_y1})-({final_x2},{final_y2})"
                    )
            else:
                logger.warning(
                    f"Invalid search range: ({x1},{y1})-({x2},{y2}). Using full frame."
                )
                # If range is invalid/too small, use full frame
                full_search = True

        # Ensure search_frame is not empty after cropping
        if search_frame is None or search_frame.size == 0:
            logger.warning("Search frame is empty after applying range.")
            return {
                "success": False,
                "error": "Invalid search range resulting in empty frame",
                "score": 0.0,
                "highest_score": 0.0,
                "search_region_x1": final_x1,
                "search_region_y1": final_y1,
                "search_region_x2": final_x2,
                "search_region_y2": final_y2,
                "search_region_full_search": full_search,
                # Add other default fields expected in the result
                "top_left_x": None,
                "top_left_y": None,
                "width": None,
                "height": None,
                "center_x": None,
                "center_y": None,
                "top_left_x_with_offset": None,
                "top_left_y_with_offset": None,
                "center_x_with_offset": None,
                "center_y_with_offset": None,
                "offset_applied_x": params.get("offset_x", 0),
                "offset_applied_y": params.get("offset_y", 0),
                "verify_wait": params.get("waitforrecheck", 0.0),
                "verify_confirmed": False,
                "verify_score": None,
                "recheck_status": "Not performed",
                "recheck_frame_timestamp": None,
                "filter_type_used": filter_type.value,
                "match_method_used": match_method.value,
                "processing_applied": None,
            }

        # Check if template is larger than search frame (OpenCV limitation)
        template_height, template_width = template.shape[:2]
        sf_height, sf_width = search_frame.shape[:2]
        if template_height > sf_height or template_width > sf_width:
            logger.warning(
                f"Template ({template_width}x{template_height}) larger than search region ({sf_width}x{sf_height})."
            )
            return {
                "success": False,
                "error": "Template larger than search region",
                "score": 0.0,
                "highest_score": 0.0,
                "search_region_x1": final_x1,
                "search_region_y1": final_y1,
                "search_region_x2": final_x2,
                "search_region_y2": final_y2,
                "search_region_full_search": full_search,
                # Add other default fields - similar to above
                "top_left_x": None,
                "top_left_y": None,
                "width": None,
                "height": None,
                "center_x": None,
                "center_y": None,
                "offset_applied_x": params.get("offset_x", 0),
                "offset_applied_y": params.get("offset_y", 0),
                "filter_type_used": filter_type.value,
                "match_method_used": match_method.value,
            }

        # Get OpenCV enum value for the match method
        cv_match_method = _get_match_method(match_method.value)

        # Perform template matching
        try:
            # Start timer for performance monitoring
            t_start = time.time()

            # 性能优化：对于大尺寸图像进行降采样，然后再进行模板匹配
            # 当图像和模板都很大时，这可以显著提高性能
            MAX_REASONABLE_SIZE = 1000 * 1000  # 100万像素
            original_search_frame = None
            original_template = None
            scale_factor = 1.0

            if search_frame.size > MAX_REASONABLE_SIZE and template.size > 1000:
                # 只缩小大图像
                original_search_frame = search_frame.copy()
                original_template = template.copy()

                # 计算合适的缩放比例，保持图像不会太小
                total_pixels = search_frame.shape[0] * search_frame.shape[1]
                scale_factor = min(
                    1.0, max(0.25, math.sqrt(MAX_REASONABLE_SIZE / total_pixels))
                )

                # 缩放图像和模板
                new_width = int(search_frame.shape[1] * scale_factor)
                new_height = int(search_frame.shape[0] * scale_factor)
                search_frame = cv2.resize(search_frame, (new_width, new_height))

                new_template_width = int(template.shape[1] * scale_factor)
                new_template_height = int(template.shape[0] * scale_factor)
                template = cv2.resize(
                    template, (new_template_width, new_template_height)
                )

                logger.debug(
                    f"Downsampled images by factor {scale_factor:.2f} for performance"
                )

            # Run the template matching operation
            result = cv2.matchTemplate(search_frame, template, cv_match_method)

            # Find the best match location
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Calculate matching score (handle method-specific scoring)
            match_score = _calculate_match_score(min_val, max_val, match_method.value)

            # Determine location based on match method
            if (
                match_method.value == MatchMethod.SqDiff
                or match_method.value == MatchMethod.SqDiffNormed
            ):
                top_left = min_loc  # For methods where smaller values are better
            else:
                top_left = max_loc  # For methods where larger values are better

            # 如果进行了降采样，需要将坐标映射回原始图像
            if scale_factor != 1.0 and original_search_frame is not None:
                # 转换回原始坐标
                top_left = (
                    int(top_left[0] / scale_factor),
                    int(top_left[1] / scale_factor),
                )
                # 恢复原始图像和模板尺寸
                search_frame = original_search_frame
                template = original_template

            # Adjust coordinates if we cropped the frame
            if not full_search:
                top_left = (top_left[0] + final_x1, top_left[1] + final_y1)

            # Calculate additional result data
            template_height, template_width = template.shape[:2]
            width, height = template_width, template_height
            center_x = top_left[0] + width // 2
            center_y = top_left[1] + height // 2

            # Apply offset if specified
            offset_x = params.get("offset_x", 0)
            offset_y = params.get("offset_y", 0)

            # Success is determined by comparing score to threshold
            success = match_score >= threshold

            # Additional logging for debug, only when needed
            if logger.isEnabledFor(logging.DEBUG):
                if success:
                    logger.debug(
                        f"Match found at ({top_left[0]},{top_left[1]}) with score {match_score:.4f} (threshold: {threshold})"
                    )
                else:
                    logger.debug(
                        f"No match found. Highest score: {match_score:.4f}, threshold: {threshold}"
                    )

            # Calculate time taken
            t_elapsed = time.time() - t_start

            # Build the result dictionary
            result_dict = {
                "success": success,
                "score": match_score,
                "highest_score": match_score,
                "top_left_x": top_left[0],
                "top_left_y": top_left[1],
                "width": width,
                "height": height,
                "center_x": center_x,
                "center_y": center_y,
                "top_left_x_with_offset": top_left[0] + offset_x,
                "top_left_y_with_offset": top_left[1] + offset_y,
                "center_x_with_offset": center_x + offset_x,
                "center_y_with_offset": center_y + offset_y,
                "offset_applied_x": offset_x,
                "offset_applied_y": offset_y,
                "verify_wait": params.get("waitforrecheck", 0.0),
                "verify_confirmed": False,
                "verify_score": None,
                "recheck_status": "Not performed",
                "recheck_frame_timestamp": None,
                "search_region_x1": final_x1,
                "search_region_y1": final_y1,
                "search_region_x2": final_x2,
                "search_region_y2": final_y2,
                "search_region_full_search": full_search,
                "filter_type_used": filter_type.value,
                "match_method_used": match_method.value,
                "processing_time": t_elapsed,
            }

            if not success:
                result_dict["error"] = "Template not found (score below threshold)"

            return result_dict

        except cv2.error as e:
            logger.error(f"OpenCV error during template matching: {e}")
            return {
                "success": False,
                "error": f"OpenCV error: {str(e)}",
                "highest_score": 0.0,
                "search_region_full_search": full_search,
                "filter_type_used": filter_type.value,
                "match_method_used": match_method.value,
            }
    except Exception as e:
        logger.exception("Unexpected error during template matching:")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "highest_score": 0.0,
        }


def _get_match_method(method_str: str) -> int:
    """
    Convert match method string to OpenCV enum value.

    Args:
        method_str: String representation of match method

    Returns:
        OpenCV method enum integer
    """
    method_mapping = {
        "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
        "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
        "ccorr_normed": cv2.TM_CCORR_NORMED,
        "ccoeff": cv2.TM_CCOEFF,
        "sqdiff": cv2.TM_SQDIFF,
        "ccorr": cv2.TM_CCORR,
    }
    return method_mapping.get(method_str.lower(), cv2.TM_CCOEFF_NORMED)


def _calculate_match_score(min_val: float, max_val: float, method: str) -> float:
    """
    Calculate normalized match score (0.0-1.0) adjusted for the matching method.
    For SQDIFF methods, lower values indicate better matches, so score is inverted.

    Args:
        min_val: Minimum match value from minMaxLoc
        max_val: Maximum match value from minMaxLoc
        method: Match method string

    Returns:
        Normalized score where higher values (closer to 1.0) indicate better matches
    """
    if method in ["sqdiff", "sqdiff_normed"]:
        # For SQDIFF methods, lower values are better
        # If min_val and max_val are equal, the transformation would be undefined
        if min_val == max_val:
            return 1.0 if min_val == 0 else 0.0

        # Normalize and invert
        if method == "sqdiff_normed":
            # For normalized methods, values are already in [0,1]
            # Invert so 1.0 is perfect match (0 becomes 1, 1 becomes 0)
            return 1.0 - min_val
        else:
            # For non-normalized methods, normalize using range
            # Be careful with division by zero
            if max_val == min_val:
                return 0.0
            # Normalize to [0,1] and invert
            return 1.0 - (min_val / max_val)
    else:
        # For all other methods (CCOEFF, CCORR), higher values are better
        if method.endswith("_normed"):
            # Already normalized to [0,1]
            return max_val
        else:
            # Normalize using range
            if max_val == 0:
                return 0.0
            return max_val / (max_val - min_val) if max_val != min_val else 1.0


def _flatten_and_reorder_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理并重新排序结果字典，确保一致的响应格式。

    Args:
        result: 原始结果字典

    Returns:
        改进格式的结果字典
    """
    # 创建结果字典，采用更高效的方式预创建
    found = result.get("success", False)
    ordered_result = {"found": found}

    # 如果找到匹配，则删除错误信息（避免矛盾状态）
    if found and "error" in result:
        # 找到匹配时不应该有错误信息
        # 避免不必要的复制，直接修改error属性
        result = result.copy()
        del result["error"]

    # 定义常用字段的固定列表，这些字段几乎总是需要的
    essential_fields = [
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
        "center_x_with_offset",
        "center_y_with_offset",
        "offset_applied_x",
        "offset_applied_y",
        "filter_type_used",
        "match_method_used",
        "frame_timestamp",
        "threshold",
        "highest_score",
    ]

    # 先添加最常用的基本字段
    for key in essential_fields:
        ordered_result[key] = result.get(key)

    # 有条件地添加其他可能的字段
    # 重新检查相关字段 - 只有在有重新检查时才添加详细信息
    if result.get("verify_wait", 0) > 0:
        recheck_fields = [
            "verify_wait",
            "verify_confirmed",
            "verify_score",
            "recheck_status",
            "recheck_frame_timestamp",
        ]
        for key in recheck_fields:
            ordered_result[key] = result.get(key)
    else:
        # 即使没有进行重新检查，也添加基本字段
        ordered_result["verify_wait"] = result.get("verify_wait", 0)
        ordered_result["verify_confirmed"] = False
        ordered_result["recheck_status"] = "Not performed"

    # 搜索区域字段
    search_fields = [
        "search_region_x1",
        "search_region_y1",
        "search_region_x2",
        "search_region_y2",
        "search_region_full_search",
    ]
    for key in search_fields:
        ordered_result[key] = result.get(key)

    # 帧信息字段
    frame_fields = ["frame_width", "frame_height"]
    for key in frame_fields:
        ordered_result[key] = result.get(key)

    # 错误信息字段 - 根据找到状态有条件地添加
    if not found or "error" in result:
        ordered_result["error"] = result.get("error")

    # 处理时间字段 - 如果存在就添加
    if "processing_time" in result:
        ordered_result["processing_time"] = result.get("processing_time")

    return ordered_result


async def execute_search(
    template_path: Path,
    params: Dict[str, Any],
    server_args: Any,  # 使用Any类型简化代码
    frame_cache: Any,  # 使用Any类型简化代码
    template_cache: Any,  # 使用Any类型简化代码
) -> Dict[str, Any]:
    """
    Execute image search with the provided parameters, using cached frame and template data.
    Runs CPU-bound template matching in a thread pool executor.

    Args:
        template_path: Path to template image
        params: Search parameters (from INI or direct)
        server_args: Server configuration arguments
        frame_cache: The FrameDataCache instance from app state
        template_cache: The TemplateCache instance from app state

    Returns:
        dict: Search results
    """
    # 确保template_path是Path对象
    template_path = Path(template_path)

    # 只在调试级别记录详细日志
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"执行图像搜索: 模板={template_path}")

    # 检查缓存中是否已有结果
    cache_key = _generate_cache_key(str(template_path), params)

    # 清理过期缓存
    _clear_expired_cache_entries()

    # 检查缓存是否命中
    if cache_key in _result_cache:
        cached_result, timestamp = _result_cache[cache_key]
        # 检查缓存是否还有效
        if time.time() - timestamp <= _RESULT_CACHE_TTL:
            # 只在调试级别记录缓存命中信息
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"缓存命中: {template_path}")
            return cached_result.copy()  # 返回缓存结果的副本

    # 根据是否需要验证步骤，决定是否使用完整结果集
    params.get("waitforrecheck", 0.0) > 0

    # 仅创建必要的最小初始结果字段
    final_result: Dict[str, Any] = {
        "success": False,
        "error": "搜索执行过程中出现内部错误",
        "template_name": template_path.name,
        "template_path": str(template_path),
        "frame_timestamp": 0.0,
        "threshold": params.get(
            "threshold", getattr(server_args, "default_threshold", 0.8)
        ),
        "verify_wait": params.get("waitforrecheck", 0.0),
    }

    try:
        # 处理搜索参数
        processed_params = params.copy()

        # 获取参数
        waitforrecheck = processed_params.get("waitforrecheck", 0.0)
        offset_x = processed_params.get("offset_x", 0)
        offset_y = processed_params.get("offset_y", 0)

        # 更新基础结果
        final_result["offset_applied_x"] = offset_x
        final_result["offset_applied_y"] = offset_y

        # 准备搜索帧（获取最新帧）
        frame_data = frame_cache.get_data()
        gray_frame = frame_data.get("gray_frame")
        frame_shape = frame_data.get("shape")
        frame_timestamp = frame_data.get("timestamp", 0.0)

        # 更新帧信息
        final_result["frame_timestamp"] = frame_timestamp
        if frame_shape:
            final_result["frame_height"] = frame_shape[0]
            final_result["frame_width"] = frame_shape[1]
            final_result["search_region_x2"] = frame_shape[1]
            final_result["search_region_y2"] = frame_shape[0]
            final_result["search_region_full_search"] = True

        # 验证帧数据
        if gray_frame is None or gray_frame.size == 0:
            logger.warning("无法获取有效的灰度帧进行匹配")
            final_result["error"] = "无法获取有效的灰度帧进行匹配"
            return _flatten_and_reorder_result(final_result)

        # 2. 准备模板和参数
        try:
            # 获取过滤器类型和参数
            filter_type_str = processed_params.get(
                "filter_type", getattr(server_args, "default_filter_type", "none")
            )
            filter_type = FilterType(filter_type_str.lower())

            # 构建过滤器参数
            filter_params = {}
            if filter_type == FilterType.CANNY:
                filter_params["canny_t1"] = processed_params.get(
                    "canny_t1", getattr(server_args, "default_canny_t1", 50)
                )
                filter_params["canny_t2"] = processed_params.get(
                    "canny_t2", getattr(server_args, "default_canny_t2", 150)
                )

            # 更新结果中的过滤器类型
            final_result["filter_type_used"] = filter_type.value

            # 加载并处理模板（使用缓存）
            (
                processed_template,
                template_width,
                template_height,
                processing_description,
            ) = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    _load_and_process_template,
                    template_path=template_path,
                    filter_type=filter_type,
                    filter_params=filter_params,
                    template_cache=template_cache,
                ),
            )

            # 更新处理信息
            final_result["processing_applied"] = processing_description

        except FilterProcessingError as e:
            logger.error(f"模板处理错误: {e}")
            final_result["error"] = f"模板处理错误: {e}"
            return _flatten_and_reorder_result(final_result)
        except Exception as e:
            logger.exception(f"模板加载或处理出错: {e}")
            final_result["error"] = f"模板加载或处理出错: {e}"
            return _flatten_and_reorder_result(final_result)

        # 3. 执行模板匹配
        try:
            # 准备匹配参数
            match_params = {
                "filter_type": filter_type.value,
                "match_method": processed_params.get(
                    "match_method",
                    getattr(server_args, "default_match_method", "ccoeff_normed"),
                ),
                "threshold": processed_params.get(
                    "threshold", getattr(server_args, "default_threshold", 0.8)
                ),
                "match_range_x1": processed_params.get("match_range_x1"),
                "match_range_y1": processed_params.get("match_range_y1"),
                "match_range_x2": processed_params.get("match_range_x2"),
                "match_range_y2": processed_params.get("match_range_y2"),
                "offset_x": offset_x,
                "offset_y": offset_y,
                "waitforrecheck": waitforrecheck,
            }

            # 通过线程池执行器运行CPU密集型匹配
            server_config = vars(server_args)
            match_result = await asyncio.get_event_loop().run_in_executor(
                None,
                functools.partial(
                    process_template_match,
                    template=processed_template,
                    frame=gray_frame,
                    params=match_params,
                    server_config=server_config,
                ),
            )

            # 合并匹配结果到最终结果
            for key, value in match_result.items():
                final_result[key] = value

            # 优化：如果不需要重新检查，并且第一次匹配成功，直接返回结果
            # 这可以节省大量时间，不必等待重新检查
            if waitforrecheck <= 0 and match_result.get("success", False):
                # 缓存成功的结果
                _result_cache[cache_key] = (final_result.copy(), time.time())
                # 返回最终格式化结果
                return _flatten_and_reorder_result(final_result)

            # 如果需要重新检查（waitforrecheck > 0），并且初始匹配成功
            if waitforrecheck > 0 and match_result.get("success", False):
                # 只在debug时记录详细日志
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"等待 {waitforrecheck}s 后执行重新检查")
                final_result["recheck_status"] = "Waiting"

                # 等待指定时间
                await asyncio.sleep(waitforrecheck)

                # 再次获取帧数据进行重新检查
                recheck_frame_data = frame_cache.get_data()
                recheck_gray_frame = recheck_frame_data.get("gray_frame")
                recheck_timestamp = recheck_frame_data.get("timestamp", 0.0)

                if recheck_gray_frame is None or recheck_gray_frame.size == 0:
                    logger.warning("重新检查失败: 无法获取有效的重新检查帧")
                    final_result["recheck_status"] = "Failed - No valid frame"
                    final_result["verify_confirmed"] = False
                    final_result["success"] = False  # 重新检查失败视为匹配失败
                else:
                    # 执行重新检查匹配
                    recheck_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        functools.partial(
                            process_template_match,
                            template=processed_template,
                            frame=recheck_gray_frame,
                            params=match_params,
                            server_config=server_config,
                        ),
                    )

                    # 更新重新检查结果
                    recheck_success = recheck_result.get("success", False)
                    recheck_score = recheck_result.get("score", 0.0)

                    final_result["recheck_status"] = (
                        "Passed" if recheck_success else "Failed"
                    )
                    final_result["verify_confirmed"] = recheck_success
                    final_result["verify_score"] = recheck_score
                    final_result["recheck_frame_timestamp"] = recheck_timestamp

                    # 重新检查失败会导致整体匹配失败
                    if not recheck_success:
                        final_result["success"] = False
                        final_result["error"] = "重新检查失败 - 得分低于阈值"

            # 缓存结果
            if final_result.get("success", False):
                _result_cache[cache_key] = (final_result.copy(), time.time())

            # 返回最终格式化结果
            formatted_result = _flatten_and_reorder_result(final_result)
            return formatted_result

        except Exception as e:
            logger.exception(f"模板匹配过程出错: {e}")
            final_result["error"] = f"匹配过程出错: {e}"
            return _flatten_and_reorder_result(final_result)

    except Exception as e:
        logger.exception("搜索执行过程中出现意外错误:")
        final_result["success"] = False
        final_result["error"] = f"搜索执行内部错误: {str(e)}"
        return _flatten_and_reorder_result(final_result)
