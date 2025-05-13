# -*- coding: utf-8 -*-
"""
错误处理测试模块
测试各种错误情况和异常处理
"""
import pytest

from modules.datatypes import InvalidParameterError  # Keeping this as it's used
from modules.datatypes import ValidationError  # Main validation error
from modules.datatypes import (
    CacheError,
    FilterProcessingError,
    ImageSearchError,
    MJPEGStreamError,
    ParameterValidator,
    TemplateLoadError,
    TemplateMatchError,
)


def test_parameter_validator_threshold():
    """测试阈值参数验证"""
    # 有效阈值
    assert ParameterValidator.validate_threshold(0.5) == 0.5
    assert ParameterValidator.validate_threshold(0.0) == 0.0
    assert ParameterValidator.validate_threshold(1.0) == 1.0

    # 无效阈值
    with pytest.raises(
        InvalidParameterError, match="Threshold must be between 0.0 and 1.0"
    ):
        ParameterValidator.validate_threshold(-0.1)
    with pytest.raises(
        InvalidParameterError, match="Threshold must be between 0.0 and 1.0"
    ):
        ParameterValidator.validate_threshold(1.1)
    with pytest.raises(InvalidParameterError, match="Threshold must be a number"):
        ParameterValidator.validate_threshold("invalid")


def test_parameter_validator_coordinates():
    """测试坐标验证"""
    # 有效坐标 (assuming image_width=100, image_height=100 for these tests)
    img_w, img_h = 100, 100
    assert ParameterValidator.validate_coordinates(10, 20, img_w, img_h) == (10, 20)
    assert ParameterValidator.validate_coordinates(0, 0, img_w, img_h) == (0, 0)
    # Valid boundary: x can be 99 if width is 100 (0-99 range)
    assert ParameterValidator.validate_coordinates(99, 99, img_w, img_h) == (99, 99)

    # Test None inputs - Now raises InvalidParameterError
    with pytest.raises(InvalidParameterError, match="Coordinate 'x' must be provided"):
        ParameterValidator.validate_coordinates(None, 20, img_w, img_h)
    with pytest.raises(InvalidParameterError, match="Coordinate 'y' must be provided"):
        ParameterValidator.validate_coordinates(10, None, img_w, img_h)
    with pytest.raises(
        InvalidParameterError, match="Coordinate 'x' must be provided"
    ):  # x checked first
        ParameterValidator.validate_coordinates(None, None, img_w, img_h)

    # 无效坐标
    with pytest.raises(
        InvalidParameterError, match="Coordinate 'x' .* cannot be negative"
    ):
        ParameterValidator.validate_coordinates(-1, 0, img_w, img_h)
    with pytest.raises(
        InvalidParameterError, match="Coordinate 'y' .* cannot be negative"
    ):
        ParameterValidator.validate_coordinates(0, -1, img_w, img_h)

    # Out of bounds (x must be < image_width)
    with pytest.raises(
        InvalidParameterError,
        match="Coordinate 'x' .* is out of bounds for image width",
    ):
        ParameterValidator.validate_coordinates(100, 0, img_w, img_h)
    with pytest.raises(
        InvalidParameterError,
        match="Coordinate 'y' .* is out of bounds for image height",
    ):
        ParameterValidator.validate_coordinates(0, 100, img_w, img_h)

    with pytest.raises(
        InvalidParameterError, match="Coordinate 'x' .* must be an integer"
    ):
        ParameterValidator.validate_coordinates("invalid", 0, img_w, img_h)
    with pytest.raises(
        InvalidParameterError, match="Coordinate 'y' .* must be an integer"
    ):
        ParameterValidator.validate_coordinates(0, "invalid", img_w, img_h)

    # Test with custom param names
    with pytest.raises(
        InvalidParameterError, match="Coordinate 'offset_x' must be provided"
    ):
        ParameterValidator.validate_coordinates(
            None, 0, img_w, img_h, param_name_x="offset_x"
        )


def test_parameter_validator_search_region():
    """测试搜索区域验证"""
    img_w, img_h = 100, 100
    # 有效区域
    assert ParameterValidator.validate_search_region(0, 0, 100, 100, img_w, img_h) == {
        "x1": 0,
        "y1": 0,
        "x2": 100,
        "y2": 100,
    }
    assert ParameterValidator.validate_search_region(10, 10, 90, 90, img_w, img_h) == {
        "x1": 10,
        "y1": 10,
        "x2": 90,
        "y2": 90,
    }

    # 全部为 None 时使用完整区域
    assert ParameterValidator.validate_search_region(
        None, None, None, None, img_w, img_h
    ) == {
        "x1": 0,
        "y1": 0,
        "x2": img_w,
        "y2": img_h,
    }

    # 无效区域 - Expect InvalidParameterError now
    with pytest.raises(
        InvalidParameterError,
        match="Search region parameter 'x1' .* cannot be negative",
    ):
        ParameterValidator.validate_search_region(-10, 0, 50, 50, img_w, img_h)

    with pytest.raises(
        InvalidParameterError,
        match="Search region coordinates must satisfy x1 < x2 and y1 < y2",
    ):
        ParameterValidator.validate_search_region(
            90, 90, 10, 10, img_w, img_h
        )  # x1 > x2
    with pytest.raises(
        InvalidParameterError,
        match="Search region coordinates must satisfy x1 < x2 and y1 < y2",
    ):
        ParameterValidator.validate_search_region(
            10, 90, 90, 10, img_w, img_h
        )  # y1 > y2

    with pytest.raises(
        InvalidParameterError, match="Search region .* exceeds frame dimensions"
    ):
        ParameterValidator.validate_search_region(
            0, 0, 110, 100, img_w, img_h
        )  # x2 too large
    with pytest.raises(
        InvalidParameterError, match="Search region .* exceeds frame dimensions"
    ):
        ParameterValidator.validate_search_region(
            0, 0, 100, 110, img_w, img_h
        )  # y2 too large

    # 部分参数为 None
    with pytest.raises(
        InvalidParameterError,
        match="Search region parameters .* must all be provided or all be omitted",
    ):
        ParameterValidator.validate_search_region(None, 0, 100, 100, img_w, img_h)
    with pytest.raises(
        InvalidParameterError,
        match="Search region parameters .* must all be provided or all be omitted",
    ):
        ParameterValidator.validate_search_region(0, None, 100, 100, img_w, img_h)
    with pytest.raises(
        InvalidParameterError,
        match="Search region parameters .* must all be provided or all be omitted",
    ):
        ParameterValidator.validate_search_region(0, 0, None, 100, img_w, img_h)
    with pytest.raises(
        InvalidParameterError,
        match="Search region parameters .* must all be provided or all be omitted",
    ):
        ParameterValidator.validate_search_region(0, 0, 100, None, img_w, img_h)

    # 类型错误
    with pytest.raises(
        InvalidParameterError,
        match="Search region parameter 'x1' .* must be an integer",
    ):
        ParameterValidator.validate_search_region("invalid", 0, 100, 100, img_w, img_h)
    with pytest.raises(
        InvalidParameterError,
        match="Search region parameter 'y1' .* must be an integer",
    ):
        ParameterValidator.validate_search_region(0, "invalid", 100, 100, img_w, img_h)


def test_parameter_validator_canny_params():
    """测试 Canny 参数验证"""
    from modules.constants import DEFAULT_CANNY_T1, DEFAULT_CANNY_T2

    # 有效参数
    assert ParameterValidator.validate_canny_params(50, 100) == (50, 100)
    assert ParameterValidator.validate_canny_params(0, 255) == (0, 255)
    # None uses defaults
    assert ParameterValidator.validate_canny_params(None, None) == (
        DEFAULT_CANNY_T1,
        DEFAULT_CANNY_T2,
    )

    # 无效参数
    with pytest.raises(
        InvalidParameterError,
        match="Canny thresholds t1 and t2 must both be provided or both be omitted",
    ):
        ParameterValidator.validate_canny_params(50, None)
    with pytest.raises(
        InvalidParameterError,
        match="Canny thresholds t1 and t2 must both be provided or both be omitted",
    ):
        ParameterValidator.validate_canny_params(None, 100)

    with pytest.raises(
        InvalidParameterError, match="Canny thresholds .* cannot be negative"
    ):
        ParameterValidator.validate_canny_params(-1, 100)
    with pytest.raises(
        InvalidParameterError, match="Canny thresholds .* cannot be negative"
    ):
        ParameterValidator.validate_canny_params(10, -100)

    with pytest.raises(
        InvalidParameterError, match="Canny threshold t2 .* cannot exceed 255"
    ):
        ParameterValidator.validate_canny_params(100, 256)

    with pytest.raises(
        InvalidParameterError, match="Canny threshold t1 .* must be less than t2"
    ):
        ParameterValidator.validate_canny_params(200, 100)  # t1 >= t2
    with pytest.raises(
        InvalidParameterError, match="Canny threshold t1 .* must be less than t2"
    ):
        ParameterValidator.validate_canny_params(100, 100)  # t1 == t2

    with pytest.raises(
        InvalidParameterError, match="Canny thresholds .* must be integers"
    ):
        ParameterValidator.validate_canny_params("invalid", 100)
    with pytest.raises(
        InvalidParameterError, match="Canny thresholds .* must be integers"
    ):
        ParameterValidator.validate_canny_params(100, "invalid")


def test_error_inheritance():
    """测试错误继承关系"""
    assert issubclass(ValidationError, ImageSearchError)
    assert issubclass(
        InvalidParameterError, ValidationError
    )  # Changed: InvalidParameterError is now a more specific ValidationError
    assert issubclass(ResourceNotFoundError, ImageSearchError)
    assert issubclass(FilterProcessingError, ImageSearchError)
    assert issubclass(TemplateLoadError, ImageSearchError)
    assert issubclass(TemplateMatchError, ImageSearchError)
    assert issubclass(MJPEGStreamError, ImageSearchError)
    assert issubclass(CacheError, ImageSearchError)


def test_error_messages_and_types():
    """测试错误消息和类型"""
    error_msg = "A test error message"
    details_dict = {"key": "value"}

    # Test base ImageSearchError
    base_error = ImageSearchError(error_msg, details=details_dict)
    assert str(base_error) == error_msg
    assert base_error.error_type == "ImageSearchError"
    assert base_error.details == details_dict

    # Test specific error types
    test_cases = [
        (ValidationError, "ValidationError"),
        (InvalidParameterError, "InvalidParameterError"),
        (ResourceNotFoundError, "ResourceNotFoundError"),
        (FilterProcessingError, "FilterProcessingError"),
        (TemplateLoadError, "TemplateLoadError"),
        (TemplateMatchError, "TemplateMatchError"),
        (MJPEGStreamError, "MJPEGStreamError"),
        (CacheError, "CacheError"),
    ]

    for ErrorClass, expected_type_name in test_cases:
        err_instance = ErrorClass(error_msg, details=details_dict)
        assert str(err_instance) == error_msg
        assert err_instance.error_type == expected_type_name
        assert err_instance.details == details_dict
        assert isinstance(err_instance, ImageSearchError)
        if ErrorClass is InvalidParameterError:
            assert isinstance(err_instance, ValidationError)

    # Test without details
    simple_error = ValidationError("Simple validation failed")
    assert str(simple_error) == "Simple validation failed"
    assert simple_error.error_type == "ValidationError"
    assert simple_error.details == {}
