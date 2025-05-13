# -*- coding: utf-8 -*-
"""
Image Processing Module
Provides template matching and image filtering functionality
"""
import logging
import os
import time  # Added time for processing_time calculation
from typing import Any, Dict, Optional

import cv2
import numpy as np

from modules.constants import (
    DEFAULT_CANNY_T1,
    DEFAULT_CANNY_T2,
    DEFAULT_FILTER_TYPE,
    DEFAULT_MATCH_METHOD,
    DEFAULT_TEMPLATE_CACHE_SIZE,
    DEFAULT_THRESHOLD,
)
from modules.datatypes import FilterType  # Ensure FilterType is imported
from modules.datatypes import MatchMethod  # Ensure MatchMethod is imported
from modules.datatypes import (
    TemplateCacheEntry,  # Ensure TemplateCacheEntry is imported for type hinting
)
from modules.datatypes import (
    FilterProcessingError,
    ParameterValidator,
    ResourceNotFoundError,
    TemplateLoadError,
    TemplateMatchError,
    ValidationError,
)
from modules.state_managers import TemplateCache

logger = logging.getLogger(__name__)


# Image Processor Class
class ImageProcessor:
    """
    处理图像处理操作，包括模板加载、预处理过滤和模板匹配。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化图像处理器

        Args:
            config: 包含默认设置的配置字典
        """
        self.config = config
        # Use default_template_cache_size from config if available, else constant
        cache_size = self.config.get("template_cache_size", DEFAULT_TEMPLATE_CACHE_SIZE)
        self.template_cache = TemplateCache(cache_size)
        logger.info(f"图像处理器已初始化 (Template Cache Size: {cache_size})")

    def _load_template(self, template_path: str) -> np.ndarray:
        """
        从磁盘加载模板图像

        Args:
            template_path: 模板图像的路径

        Returns:
            模板图像的 numpy 数组

        Raises:
            ResourceNotFoundError: 当模板文件不存在时
            TemplateLoadError: 当模板加载或解码失败时
        """
        try:
            # 验证图像格式 - validate_image_format now returns the path if valid
            validated_template_path = ParameterValidator.validate_image_format(
                template_path
            )

            if not os.path.exists(validated_template_path):
                raise ResourceNotFoundError(
                    message=f"未找到模板文件: {validated_template_path}",
                    details={"path": validated_template_path},
                )

            # 使用 imdecode 进行健壮的路径处理
            with open(validated_template_path, "rb") as f_img:
                img_bytes = f_img.read()

            # 使用 OpenCV 将图像字节直接解码为灰度图
            nparr = np.frombuffer(img_bytes, np.uint8)
            template = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if template is None:
                raise TemplateLoadError(
                    message=f"解码模板图像失败: {validated_template_path}",
                    details={
                        "path": validated_template_path,
                        "error": "cv2.imdecode 返回 None",
                    },
                )

            return template

        except (
            ValidationError,
            ResourceNotFoundError,
        ):  # Re-raise specific validation/not found errors
            raise
        except Exception as e:  # Catch other errors during loading
            raise TemplateLoadError(
                message=f"加载模板时出错: {template_path}",  # Use original path in message
                details={
                    "path": template_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    def _preprocess_image(
        self,
        image: np.ndarray,
        filter_type_str: str,  # Renamed to avoid conflict with FilterType enum
        canny_t1: Optional[int] = None,  # Allow None for default handling
        canny_t2: Optional[int] = None,  # Allow None for default handling
    ) -> np.ndarray:
        """
        对图像应用预处理过滤器

        Args:
            image: 输入图像（灰度或彩色，将转换为灰度）
            filter_type_str: 过滤器类型 ('none' 或 'canny')
            canny_t1: Canny 边缘检测的低阈值 (使用模块级默认值如果None)
            canny_t2: Canny 边缘检测的高阈值 (使用模块级默认值如果None)

        Returns:
            处理后的图像 (灰度)

        Raises:
            ValidationError: 当参数验证失败时
            FilterProcessingError: 当图像处理失败时
        """
        try:
            # 验证过滤器类型
            filter_type_enum = FilterType.validate(filter_type_str)

            # 确保图像是灰度图
            if len(image.shape) == 3:  # Color image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 2:  # Already grayscale
                gray = image.copy()  # Work on a copy
            else:
                raise FilterProcessingError(
                    message=f"Unsupported image shape for preprocessing: {image.shape}",
                    details={"shape": image.shape},
                )

            # 应用过滤器
            if filter_type_enum == FilterType.CANNY:
                # Validate Canny params, using constants if None are provided
                # ParameterValidator.validate_canny_params handles None by using constants
                # It also validates if t1/t2 are provided together.
                # If canny_t1/t2 are passed as None here, validate_canny_params will use DEFAULT_CANNY_T1/T2
                validated_t1, validated_t2 = ParameterValidator.validate_canny_params(
                    canny_t1, canny_t2
                )
                try:
                    return cv2.Canny(gray, validated_t1, validated_t2)
                except cv2.error as e_cv:  # Catch specific OpenCV errors
                    raise FilterProcessingError(
                        message=f"应用 Canny 过滤器失败 (cv2.error): {e_cv}",
                        details={
                            "t1": validated_t1,
                            "t2": validated_t2,
                            "error": str(e_cv),
                        },
                    ) from e_cv
                except Exception as e:  # Catch other unexpected errors during Canny
                    raise FilterProcessingError(
                        message=f"应用 Canny 过滤器失败: {e}",
                        details={
                            "t1": validated_t1,
                            "t2": validated_t2,
                            "error": str(e),
                        },
                    ) from e
            else:  # FilterType.NONE
                return gray

        except (
            ValidationError
        ):  # Re-raise validation errors from FilterType.validate or validate_canny_params
            raise
        except FilterProcessingError:  # Re-raise if already a FilterProcessingError
            raise
        except Exception as e:  # Catch other unexpected errors
            raise FilterProcessingError(
                message="图像预处理失败",
                details={
                    "filter_type": filter_type_str,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    def _get_match_method_cv2(self, method_str: str) -> int:
        """
        获取 OpenCV 匹配方法常量

        Args:
            method_str: 匹配方法名称

        Returns:
            OpenCV 匹配方法常量

        Raises:
            ValidationError: 当匹配方法无效时
        """
        # Validate method_str using the enum's validator first
        try:
            match_method_enum = MatchMethod.validate(method_str)
        except (
            ValueError
        ) as e:  # MatchMethod.validate raises ValueError for invalid methods
            raise ValidationError(
                message=f"无效的匹配方法字符串: {method_str}. Error: {e}",
                details={"provided": method_str},
            ) from e

        # Map enum member value to cv2 constant
        # This map should be comprehensive and align with MatchMethod enum values
        # and constants.CV2_MATCH_METHODS_MAP
        from modules.constants import (
            CV2_MATCH_METHODS_MAP,  # Local import for clarity or use a class member
        )

        cv2_method_val = CV2_MATCH_METHODS_MAP.get(match_method_enum.value)

        if cv2_method_val is None:
            # This case should ideally not be reached if MatchMethod enum and CV2_MATCH_METHODS_MAP are consistent
            raise ValidationError(
                message=f"内部错误: 匹配方法 '{match_method_enum.value}' 在 CV2_MATCH_METHODS_MAP 中没有对应的 OpenCV 常量。",
                details={"method_value": match_method_enum.value},
            )
        return cv2_method_val

    def _calculate_score(self, match_val: float, method_str: str) -> float:
        """
        Calculates a normalized match score.
        For TM_SQDIFF_NORMED, it inverts the value (0 is best -> 1 is best).
        For TM_CCOEFF_NORMED, TM_CCORR_NORMED, it returns the value directly as they range [-1, 1] (1 is best).
        Other methods might need specific handling if used for thresholding.

        Args:
            match_val: Raw match value from cv2.matchTemplate.
            method_str: The match method string (e.g., "ccoeff_normed").

        Returns:
            A score. For "sqdiff_normed", higher is better [0,1].
            For "ccoeff_normed", "ccorr_normed", score is in [-1,1], higher is better.
        """
        try:
            # Validate method_str to get its canonical form (enum value)
            # This ensures method_lower is based on a validated method string
            validated_method_enum_val = MatchMethod.validate(method_str).value
        except (
            ValueError
        ) as e:  # Should not happen if method_str comes from a validated source
            raise ValidationError(
                f"Invalid method string '{method_str}' for score calculation: {e}"
            )

        if (
            validated_method_enum_val == MatchMethod.SqDiffNormed.value
        ):  # "sqdiff_normed"
            # TM_SQDIFF_NORMED: 0 is perfect match, 1 is worst. We want higher is better.
            return 1.0 - match_val
        elif validated_method_enum_val in [
            MatchMethod.CCoeffNormed.value,
            MatchMethod.CCorrNormed.value,
        ]:
            # TM_CCOEFF_NORMED, TM_CCORR_NORMED: Result is in [-1, 1]. 1 is perfect match.
            return match_val
        # Handling for other methods if they are expected to be used with thresholding:
        # For non-normalized methods like TM_SQDIFF, TM_CCOEFF, TM_CCORR, their raw 'match_val'
        # might not be directly comparable as a score between 0 and 1 or -1 and 1.
        # If such methods are used, this function or its callers need a clear contract on scoring.
        # For now, we assume primarily normalized methods are used for direct scoring.
        # If other methods are used and `match_val` is directly comparable:
        # return match_val # or max(0.0, match_val) if they are always positive and higher is better.
        # This part requires careful consideration based on expected usage of all supported match methods.
        # The original code did `max(0.0, match_val)` for non-sqdiff_normed.
        # Given that threshold is usually >0, `max(0.0, ...)` makes sense if raw values are always positive or if negative means "no match".
        # However, for ccoeff_normed, -1 is a strong negative correlation.
        # Let's stick to direct values for ccoeff_normed/ccorr_normed as they are meaningful.
        # For other methods, if used, their scoring logic needs to be defined.
        # Assuming if other methods are passed, their `match_val` is already a "higher is better" score.
        logger.warning(
            f"Calculating score for method '{validated_method_enum_val}' which does not have explicit score adjustment. Returning raw match_val: {match_val}"
        )
        return match_val

    def _get_template(
        self,
        template_path: str,
        filter_type_str: str,
        canny_t1: Optional[int] = None,
        canny_t2: Optional[int] = None,
    ) -> TemplateCacheEntry:  # Return type changed to TemplateCacheEntry
        """
        从缓存获取处理过的模板，如果不存在则加载并处理。

        Args:
            template_path: 模板图像的路径
            filter_type_str: 过滤器类型字符串
            canny_t1: Canny 边缘检测的低阈值
            canny_t2: Canny 边缘检测的高阈值

        Returns:
            A TemplateCacheEntry named tuple containing (processed_template, width, height, processing_details_str)

        Raises:
            ValidationError: 当参数验证失败时
            ResourceNotFoundError: 当模板文件不存在时
            TemplateLoadError: 当模板加载失败时
            FilterProcessingError: 当图像处理失败时
        """
        try:
            # 验证并转换 filter_type 字符串为 FilterType 枚举
            filter_type_enum = FilterType.validate(filter_type_str)

            # 准备过滤器参数 for cache key and processing
            # These will be validated/defaulted by _preprocess_image if actually used
            filter_params_for_cache_key: Dict[str, Any] = {}
            effective_canny_t1 = canny_t1
            effective_canny_t2 = canny_t2

            if filter_type_enum == FilterType.CANNY:
                # Use provided Canny params or fall back to defaults from config/constants
                # ParameterValidator.validate_canny_params will give defaults if None
                validated_t1, validated_t2 = ParameterValidator.validate_canny_params(
                    canny_t1, canny_t2
                )
                effective_canny_t1 = validated_t1
                effective_canny_t2 = validated_t2
                filter_params_for_cache_key["canny_t1"] = effective_canny_t1
                filter_params_for_cache_key["canny_t2"] = effective_canny_t2

            # 从缓存获取
            cached_entry = self.template_cache.get_template(
                template_path, filter_type_enum, filter_params_for_cache_key
            )

            if cached_entry is not None:
                logger.debug(
                    f"模板 {template_path} (filter: {filter_type_str}, params: {filter_params_for_cache_key}) 从缓存加载"
                )
                return cached_entry  # cached_entry is already a TemplateCacheEntry

            # 加载并处理模板
            logger.debug(
                f"模板 {template_path} (filter: {filter_type_str}, params: {filter_params_for_cache_key}) 缓存未命中，正在处理..."
            )
            raw_template = self._load_template(
                template_path
            )  # Can raise ResourceNotFoundError, TemplateLoadError

            processed_template = self._preprocess_image(  # Can raise FilterProcessingError, ValidationError
                raw_template,
                filter_type_enum.value,  # Pass enum value string
                effective_canny_t1,  # Pass effective (validated or None if not Canny) params
                effective_canny_t2,
            )

            height, width = processed_template.shape[:2]

            # Create a string summary of processing for the cache entry
            processing_details_str = f"filter:{filter_type_enum.value}"
            if filter_type_enum == FilterType.CANNY:
                processing_details_str += (
                    f",canny_t1:{effective_canny_t1},canny_t2:{effective_canny_t2}"
                )

            template_entry = TemplateCacheEntry(
                processed_template=processed_template,
                width=width,
                height=height,
                processing_details=processing_details_str,  # Store how it was processed
            )

            # 存入缓存
            self.template_cache.store_template(
                template_path,
                filter_type_enum,
                filter_params_for_cache_key,
                template_entry,
            )

            return template_entry

        except (
            ValidationError,
            ResourceNotFoundError,
            TemplateLoadError,
            FilterProcessingError,
        ):
            raise  # Re-raise known specific errors
        except (
            Exception
        ) as e:  # Catch any other unexpected error during template retrieval/processing
            logger.exception(
                f"获取或处理模板时发生未知错误: {template_path}, filter: {filter_type_str}, error: {e}"
            )
            # Wrap in a more generic error like TemplateLoadError or a new GetTemplateError
            raise TemplateLoadError(
                message=f"获取或处理模板时发生未知错误: {template_path}",
                details={
                    "template_path": template_path,
                    "filter_type": filter_type_str,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    def match_template(
        self, frame: np.ndarray, template_path: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        在帧中搜索模板图像

        Args:
            frame: 要搜索的帧图像
            template_path: 模板图像的路径
            params: 匹配参数字典，包括：
                - filter_type: 过滤器类型 ('none' 或 'canny')
                - match_method: 匹配方法
                - threshold: 匹配阈值
                - x1, y1, x2, y2: 搜索区域坐标
                - offsetx, offsety: 结果坐标偏移
                - canny_t1, canny_t2: Canny 边缘检测参数

        Returns:
            匹配结果字典

        Raises:
            ValidationError: 当参数验证失败时
            ResourceNotFoundError: 当模板文件不存在时
            TemplateLoadError: 当模板加载失败时
            FilterProcessingError: 当图像处理失败时
            TemplateMatchError: 当模板匹配失败时
        """
        processing_start_time = time.time()
        try:
            # Extract and validate parameters with defaults from config or constants
            # Default values from self.config (loaded from server.ini) or fallback to constants.py defaults

            # Filter Type
            default_ft = self.config.get("default_filter_type", DEFAULT_FILTER_TYPE)
            filter_type_str = params.get("filter_type", default_ft)
            # filter_type_enum will be validated by _get_template or _preprocess_image

            # Match Method
            default_mm = self.config.get("default_match_method", DEFAULT_MATCH_METHOD)
            match_method_str = params.get("match_method", default_mm)
            # match_method_enum will be validated by _get_match_method_cv2

            # Threshold
            default_thresh = self.config.get("default_threshold", DEFAULT_THRESHOLD)
            threshold_val = ParameterValidator.validate_threshold(
                params.get("threshold", default_thresh)
            )

            # Canny Parameters
            # These are passed to _get_template and _preprocess_image, which will use defaults
            # from ParameterValidator.validate_canny_params if these are None.
            # The defaults in ParameterValidator.validate_canny_params come from constants.py.
            # If server.ini specifies default_canny_t1/t2, those should be used by ParameterValidator.
            # This requires ParameterValidator to be aware of self.config or constants to be updated by config.
            # For now, ParameterValidator uses constants directly.
            # So, if server.ini defines default_canny_t1/t2, they should be read into these variables
            # if params.get("canny_t1/t2") are None.

            default_canny_t1 = self.config.get("default_canny_t1", DEFAULT_CANNY_T1)
            default_canny_t2 = self.config.get("default_canny_t2", DEFAULT_CANNY_T2)

            canny_t1_param = params.get("canny_t1")
            canny_t2_param = params.get("canny_t2")

            # Effective Canny params for this specific match operation
            # ParameterValidator.validate_canny_params will use its own defaults (from constants.py)
            # if canny_t1_param or canny_t2_param is None.
            # If we want server.ini defaults (via self.config) to take precedence over constants.py defaults
            # when user provides no Canny params, we need to pass them here.

            # Logic: User params > self.config defaults > constants.py defaults
            # ParameterValidator.validate_canny_params uses constants.py defaults if None is passed.
            # So, if user passes None, we should pass self.config's defaults to validator.

            eff_c1 = canny_t1_param if canny_t1_param is not None else default_canny_t1
            eff_c2 = canny_t2_param if canny_t2_param is not None else default_canny_t2

            # Now validate_canny_params will either use user-provided (eff_c1, eff_c2 if not None)
            # or config-derived (eff_c1, eff_c2 from default_canny_t1/t2)
            # or fall back to its own internal defaults if even config-derived are None (which they are not here).
            # This step is implicitly handled by _get_template and _preprocess_image.
            # We just need to pass eff_c1 and eff_c2.

            # Get processed template (from cache or by loading and processing)
            # _get_template handles validation of filter_type_str and Canny params internally
            template_entry = self._get_template(
                template_path,
                filter_type_str,
                eff_c1,
                eff_c2,
            )
            processed_template = template_entry.processed_template
            template_width = template_entry.width
            template_height = template_entry.height

            # Preprocess frame image (apply the same filter as template)
            # _preprocess_image also handles Canny param validation/defaulting using eff_c1, eff_c2
            processed_frame = self._preprocess_image(
                frame,
                filter_type_str,  # Pass the original string, it will be validated
                eff_c1,
                eff_c2,
            )

            # Validate and get search region
            frame_height, frame_width = processed_frame.shape[:2]
            search_region = ParameterValidator.validate_search_region(
                params.get("x1"),
                params.get("y1"),
                params.get("x2"),
                params.get("y2"),
                frame_width,
                frame_height,
            )

            # Extract search region (ROI)
            roi = processed_frame[
                search_region["y1"] : search_region["y2"],
                search_region["x1"] : search_region["x2"],
            ]

            # Validate ROI size against template size
            if roi.shape[0] < template_height or roi.shape[1] < template_width:
                raise TemplateMatchError(  # Changed from ValidationError for more specific error context
                    message="搜索区域小于模板尺寸",
                    details={
                        "roi_size": (roi.shape[1], roi.shape[0]),  # W, H
                        "template_size": (template_width, template_height),  # W, H
                        "search_region": search_region,
                    },
                )

            # Perform template matching
            try:
                cv2_method = self._get_match_method_cv2(match_method_str)
                match_results_matrix = cv2.matchTemplate(
                    roi, processed_template, cv2_method
                )
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_results_matrix)
            except cv2.error as e_cv:
                raise TemplateMatchError(
                    message=f"模板匹配失败 (cv2.error): {e_cv}",
                    details={"method": match_method_str, "error": str(e_cv)},
                ) from e_cv
            except (
                Exception
            ) as e:  # Catch other errors during matchTemplate or minMaxLoc
                raise TemplateMatchError(
                    message=f"模板匹配失败: {e}",
                    details={"method": match_method_str, "error": str(e)},
                ) from e

            # Determine best match value and location based on method
            actual_match_value = (
                min_val
                if MatchMethod.validate(match_method_str).value
                in [MatchMethod.SqDiff.value, MatchMethod.SqDiffNormed.value]
                else max_val
            )
            best_match_location = (
                min_loc
                if MatchMethod.validate(match_method_str).value
                in [MatchMethod.SqDiff.value, MatchMethod.SqDiffNormed.value]
                else max_loc
            )

            # Calculate score using our scoring logic
            score = self._calculate_score(actual_match_value, match_method_str)

            # Calculate center point of the found template in ROI coordinates
            center_x_roi = best_match_location[0] + template_width // 2
            center_y_roi = best_match_location[1] + template_height // 2

            # Convert to full frame coordinates by adding ROI offset
            center_x_frame = center_x_roi + search_region["x1"]
            center_y_frame = center_y_roi + search_region["y1"]

            # Apply user-specified offset
            offset_x = params.get("offsetx", 0)
            offset_y = params.get("offsety", 0)
            final_center_x = center_x_frame + offset_x
            final_center_y = center_y_frame + offset_y

            top_left_x = final_center_x - template_width // 2
            top_left_y = final_center_y - template_height // 2

            # Determine success based on threshold
            # Note: _calculate_score ensures for ccoeff_normed, score is [-1,1].
            # Threshold should be set accordingly by user.
            success = score >= threshold_val

            result_dict = {
                "success": success,
                "template_path": template_path,  # Add template path to result
                "score": score,
                "threshold": threshold_val,
                "center_coords": (final_center_x, final_center_y),
                "top_left_coords": (top_left_x, top_left_y),
                "size": (template_width, template_height),
                "offset_coords": (
                    offset_x,
                    offset_y,
                ),  # The user-provided offset that was applied
                "processing_details": {
                    "filter_type_used": filter_type_str,  # Actual filter string used
                    "match_method_used": match_method_str,  # Actual method string used
                    "search_region_used": search_region,
                    "user_offset_applied": {"x": offset_x, "y": offset_y},
                    "processing_time_seconds": time.time() - processing_start_time,
                    **(
                        {
                            "canny_params_used": {"t1": eff_c1, "t2": eff_c2}
                        }  # Use validated/defaulted Canny params
                        if FilterType.validate(filter_type_str) == FilterType.CANNY
                        else {}
                    ),
                },
                # "frame_info" will be added by the API layer
            }
            return result_dict

        except (
            ValidationError,
            ResourceNotFoundError,
            TemplateLoadError,
            FilterProcessingError,
            TemplateMatchError,
        ) as e:
            # Re-raise known errors
            logger.warning(
                f"Known error during template matching for {template_path}: {e.error_type if hasattr(e, 'error_type') else type(e).__name__} - {e}"
            )
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.exception(f"模板匹配时发生意外错误 for {template_path}: {e}")
            raise TemplateMatchError(
                message="模板匹配过程中发生意外错误",
                details={
                    "template_path": template_path,
                    "params": params,  # Log params for debugging
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e
