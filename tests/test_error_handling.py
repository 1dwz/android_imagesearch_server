# -*- coding: utf-8 -*-
"""
错误处理测试模块
测试各种错误情况和异常处理
"""
import pytest

from modules.datatypes import (
    CacheError,
    FilterProcessingError,
    ImageSearchError,
    InvalidParameterError,
    MJPEGStreamError,
    ParameterValidator,
    TemplateLoadError,
    TemplateMatchError,
    ValidationError,
)


def test_parameter_validator_threshold():
    """测试阈值参数验证"""
    # 有效阈值
    assert ParameterValidator.validate_threshold(0.5) == 0.5
    assert ParameterValidator.validate_threshold(0.0) == 0.0
    assert ParameterValidator.validate_threshold(1.0) == 1.0

    # 无效阈值
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_threshold(-0.1)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_threshold(1.1)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_threshold("invalid")


def test_parameter_validator_coordinates():
    """测试坐标验证"""
    # 有效坐标
    assert ParameterValidator.validate_coordinates(10, 20, 100, 100) == (10, 20)
    assert ParameterValidator.validate_coordinates(0, 0, 100, 100) == (0, 0)
    assert ParameterValidator.validate_coordinates(99, 99, 100, 100) == (99, 99)

    # 无效坐标
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_coordinates(-1, 0, 100, 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_coordinates(0, -1, 100, 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_coordinates(100, 0, 100, 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_coordinates(0, 100, 100, 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_coordinates(None, 0, 100, 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_coordinates(0, None, 100, 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_coordinates("invalid", 0, 100, 100)


def test_parameter_validator_search_region():
    """测试搜索区域验证"""
    # 有效区域
    assert ParameterValidator.validate_search_region(0, 0, 100, 100, 100, 100) == {
        "x1": 0,
        "y1": 0,
        "x2": 100,
        "y2": 100,
    }
    assert ParameterValidator.validate_search_region(10, 10, 90, 90, 100, 100) == {
        "x1": 10,
        "y1": 10,
        "x2": 90,
        "y2": 90,
    }

    # 全部为 None 时使用完整区域
    assert ParameterValidator.validate_search_region(
        None, None, None, None, 100, 100
    ) == {
        "x1": 0,
        "y1": 0,
        "x2": 100,
        "y2": 100,
    }

    # 无效区域 - 期望 ValidationError
    with pytest.raises(ValidationError, match="搜索区域参数不能为负数"):
        ParameterValidator.validate_search_region(-10, 0, 50, 50, 100, 100)

    with pytest.raises(
        ValidationError, match="搜索区域参数必须满足: x1 < x2 且 y1 < y2"
    ):
        ParameterValidator.validate_search_region(90, 90, 10, 10, 100, 100)

    with pytest.raises(ValidationError, match="搜索区域超出图像范围"):
        ParameterValidator.validate_search_region(0, 0, 110, 110, 100, 100)

    # 部分参数为 None
    with pytest.raises(ValidationError, match="搜索区域参数必须全部提供或全部省略"):
        ParameterValidator.validate_search_region(None, 0, 100, 100, 100, 100)

    # 类型错误
    with pytest.raises(ValidationError, match="搜索区域参数必须是整数"):
        ParameterValidator.validate_search_region("invalid", 0, 100, 100, 100, 100)


def test_parameter_validator_canny_params():
    """测试 Canny 参数验证"""
    # 有效参数
    assert ParameterValidator.validate_canny_params(100, 200) == (100, 200)
    assert ParameterValidator.validate_canny_params(0, 255) == (0, 255)

    # 无效参数
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_canny_params(-1, 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_canny_params(100, 256)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_canny_params(200, 100)  # t1 >= t2
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_canny_params("invalid", 100)
    with pytest.raises(InvalidParameterError):
        ParameterValidator.validate_canny_params(100, "invalid")


def test_error_inheritance():
    """测试错误继承关系"""
    # 所有错误类型都应该继承自 ImageSearchError
    assert issubclass(FilterProcessingError, ImageSearchError)
    assert issubclass(TemplateLoadError, ImageSearchError)
    assert issubclass(TemplateMatchError, ImageSearchError)
    assert issubclass(InvalidParameterError, ImageSearchError)
    assert issubclass(MJPEGStreamError, ImageSearchError)
    assert issubclass(CacheError, ImageSearchError)


def test_error_messages():
    """测试错误消息"""
    # 测试错误消息格式
    error_msg = "测试错误消息"

    filter_error = FilterProcessingError(error_msg)
    assert str(filter_error) == error_msg
    assert isinstance(filter_error, ImageSearchError)

    template_error = TemplateLoadError(error_msg)
    assert str(template_error) == error_msg
    assert isinstance(template_error, ImageSearchError)

    match_error = TemplateMatchError(error_msg)
    assert str(match_error) == error_msg
    assert isinstance(match_error, ImageSearchError)

    param_error = InvalidParameterError(error_msg)
    assert str(param_error) == error_msg
    assert isinstance(param_error, ImageSearchError)

    stream_error = MJPEGStreamError(error_msg)
    assert str(stream_error) == error_msg
    assert isinstance(stream_error, ImageSearchError)

    cache_error = CacheError(error_msg)
    assert str(cache_error) == error_msg
    assert isinstance(cache_error, ImageSearchError)
