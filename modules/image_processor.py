# -*- coding: utf-8 -*-
"""
Image Processing Module
Provides template matching and image filtering functionality
"""
import logging
import os
from typing import Any, Dict, Optional, Tuple

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
from modules.datatypes import (
    FilterProcessingError,
    FilterType,
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
        self.template_cache = TemplateCache(
            config.get("template_cache_size", DEFAULT_TEMPLATE_CACHE_SIZE)
        )
        logger.info("图像处理器已初始化")

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
            # 验证图像格式
            template_path = ParameterValidator.validate_image_format(template_path)

            if not os.path.exists(template_path):
                raise ResourceNotFoundError(
                    message=f"未找到模板文件: {template_path}",
                    details={"path": template_path},
                )

            # 使用 imdecode 进行健壮的路径处理
            with open(template_path, "rb") as f_img:
                img_bytes = f_img.read()

            # 使用 OpenCV 将图像字节直接解码为灰度图
            nparr = np.frombuffer(img_bytes, np.uint8)
            template = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if template is None:
                raise TemplateLoadError(
                    message=f"解码模板图像失败: {template_path}",
                    details={"path": template_path, "error": "cv2.imdecode 返回 None"},
                )

            return template

        except (ValidationError, ResourceNotFoundError):
            raise
        except Exception as e:
            raise TemplateLoadError(
                message=f"加载模板时出错: {template_path}",
                details={
                    "path": template_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _preprocess_image(
        self,
        image: np.ndarray,
        filter_type: str,
        canny_t1: int = DEFAULT_CANNY_T1,
        canny_t2: int = DEFAULT_CANNY_T2,
    ) -> np.ndarray:
        """
        对图像应用预处理过滤器

        Args:
            image: 输入图像（灰度）
            filter_type: 过滤器类型 ('none' 或 'canny')
            canny_t1: Canny 边缘检测的低阈值
            canny_t2: Canny 边缘检测的高阈值

        Returns:
            处理后的图像

        Raises:
            ValidationError: 当参数验证失败时
            FilterProcessingError: 当图像处理失败时
        """
        try:
            # 验证过滤器类型
            filter_type = FilterType.validate(filter_type)

            # 确保图像是灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # 应用过滤器
            if filter_type == FilterType.CANNY:
                # 验证 Canny 参数
                t1, t2 = ParameterValidator.validate_canny_params(canny_t1, canny_t2)
                try:
                    return cv2.Canny(gray, t1, t2)
                except Exception as e:
                    raise FilterProcessingError(
                        message="应用 Canny 过滤器失败",
                        details={"t1": t1, "t2": t2, "error": str(e)},
                    )
            else:
                return gray

        except ValidationError:
            raise
        except Exception as e:
            raise FilterProcessingError(
                message="图像预处理失败",
                details={
                    "filter_type": filter_type,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

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
        method_map = {
            "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
            "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
            "ccorr_normed": cv2.TM_CCORR_NORMED,
            "sqdiff": cv2.TM_SQDIFF,
            "ccoeff": cv2.TM_CCOEFF,
            "ccorr": cv2.TM_CCORR,
        }

        try:
            method = method_map.get(method_str.lower())
            if method is None:
                raise ValidationError(
                    message=f"无效的匹配方法: {method_str}",
                    details={
                        "provided": method_str,
                        "valid_methods": list(method_map.keys()),
                    },
                )
            return method
        except AttributeError:
            raise ValidationError(
                message="匹配方法必须是字符串类型",
                details={"provided_type": type(method_str).__name__},
            )

    def _calculate_score(self, match_val: float, method_str: str) -> float:
        """
        计算归一化的匹配分数 (0.0-1.0)

        Args:
            match_val: cv2.matchTemplate 的原始匹配值
            method_str: 使用的匹配方法

        Returns:
            归一化分数（越高越好）
        """
        try:
            if method_str.lower() == "sqdiff_normed":
                return 1.0 - match_val  # 反转使得越高越好
            return max(0.0, match_val)  # 确保非负
        except Exception as e:
            raise ValidationError(
                message="计算匹配分数失败",
                details={"match_val": match_val, "method": method_str, "error": str(e)},
            )

    def _get_template(
        self,
        template_path: str,
        filter_type_str: str,
        canny_t1: Optional[int] = None,
        canny_t2: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, int]:
        """
        从缓存获取处理过的模板，如果不存在则加载并处理。

        Args:
            template_path: 模板图像的路径
            filter_type_str: 过滤器类型字符串
            canny_t1: Canny 边缘检测的低阈值
            canny_t2: Canny 边缘检测的高阈值

        Returns:
            处理过的模板数据元组 (processed_template, width, height)

        Raises:
            ValidationError: 当参数验证失败时
            ResourceNotFoundError: 当模板文件不存在时
            TemplateLoadError: 当模板加载失败时
            FilterProcessingError: 当图像处理失败时
        """
        try:
            # 验证并转换 filter_type 字符串为 FilterType 枚举
            filter_type_enum = FilterType.validate(filter_type_str)

            # 准备过滤器参数
            filter_params = {}
            if filter_type_enum == FilterType.CANNY:
                # 使用默认值或提供的值
                c1_eff = (
                    canny_t1
                    if canny_t1 is not None
                    else self.config.get("default_canny_t1", DEFAULT_CANNY_T1)
                )
                c2_eff = (
                    canny_t2
                    if canny_t2 is not None
                    else self.config.get("default_canny_t2", DEFAULT_CANNY_T2)
                )
                filter_params["canny_t1"] = c1_eff
                filter_params["canny_t2"] = c2_eff

            # 从缓存获取
            cached_template = self.template_cache.get_template(
                template_path, filter_type_enum, filter_params
            )

            if cached_template is not None:
                logger.debug(
                    f"模板 {template_path} (filter: {filter_type_str}) 从缓存加载"
                )
                return cached_template

            # 加载并处理模板
            logger.debug(
                f"模板 {template_path} (filter: {filter_type_str}) 缓存未命中，正在处理..."
            )
            template = self._load_template(template_path)
            processed = self._preprocess_image(
                template,
                filter_type_str,
                filter_params.get("canny_t1"),
                filter_params.get("canny_t2"),
            )

            # 获取尺寸
            height, width = processed.shape[:2]

            # 存入缓存
            template_data = (processed, width, height)
            self.template_cache.store_template(
                template_path, filter_type_enum, filter_params, template_data
            )

            return template_data

        except Exception as e:
            if isinstance(
                e,
                (
                    ValidationError,
                    ResourceNotFoundError,
                    TemplateLoadError,
                    FilterProcessingError,
                ),
            ):
                raise
            raise TemplateLoadError(
                message="处理模板时发生未知错误",
                details={
                    "template_path": template_path,
                    "filter_type": filter_type_str,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

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
        try:
            # 提取和验证参数
            filter_type = params.get("filter_type", DEFAULT_FILTER_TYPE)
            match_method = params.get("match_method", DEFAULT_MATCH_METHOD)
            threshold = ParameterValidator.validate_threshold(
                params.get("threshold", DEFAULT_THRESHOLD)
            )

            # 获取处理后的模板
            template, template_width, template_height = self._get_template(
                template_path,
                filter_type,
                params.get("canny_t1"),
                params.get("canny_t2"),
            )

            # 预处理帧图像
            processed_frame = self._preprocess_image(
                frame,
                filter_type,
                params.get("canny_t1"),
                params.get("canny_t2"),
            )

            # 验证并获取搜索区域
            frame_height, frame_width = processed_frame.shape[:2]
            search_region = ParameterValidator.validate_search_region(
                params.get("x1"),
                params.get("y1"),
                params.get("x2"),
                params.get("y2"),
                frame_width,
                frame_height,
            )

            # 提取搜索区域
            roi = processed_frame[
                search_region["y1"] : search_region["y2"],
                search_region["x1"] : search_region["x2"],
            ]

            # 验证 ROI 大小
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                raise ValidationError(
                    message="搜索区域小于模板尺寸",
                    details={
                        "roi_size": roi.shape[:2],
                        "template_size": template.shape[:2],
                    },
                )

            # 执行模板匹配
            try:
                cv2_method = self._get_match_method_cv2(match_method)
                result = cv2.matchTemplate(roi, template, cv2_method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            except Exception as e:
                raise TemplateMatchError(
                    message="模板匹配失败",
                    details={"method": match_method, "error": str(e)},
                )

            # 计算匹配分数和位置
            match_val = min_val if match_method == "sqdiff_normed" else max_val
            score = self._calculate_score(match_val, match_method)

            # 获取最佳匹配位置
            best_loc = min_loc if match_method == "sqdiff_normed" else max_loc

            # 计算中心点和偏移
            x = best_loc[0] + template_width // 2
            y = best_loc[1] + template_height // 2

            # 添加搜索区域偏移
            x += search_region["x1"]
            y += search_region["y1"]

            # 添加用户指定的偏移
            offsetx = params.get("offsetx", 0)
            offsety = params.get("offsety", 0)
            x += offsetx
            y += offsety

            # 构建结果
            success = score >= threshold
            result = {
                "success": success,
                "score": score,
                "threshold": threshold,
                "center_x": x,
                "center_y": y,
                "top_left_x": x - template_width // 2,
                "top_left_y": y - template_height // 2,
                "width": template_width,
                "height": template_height,
                "processing_details": {
                    "filter_type": filter_type,
                    "match_method": match_method,
                    "search_region": search_region,
                    "offset": {"x": offsetx, "y": offsety},
                },
            }

            return result

        except Exception as e:
            if isinstance(
                e,
                (
                    ValidationError,
                    ResourceNotFoundError,
                    TemplateLoadError,
                    FilterProcessingError,
                    TemplateMatchError,
                ),
            ):
                raise
            raise TemplateMatchError(
                message="模板匹配过程中发生未知错误",
                details={
                    "template_path": template_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
