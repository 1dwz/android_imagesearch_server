# -*- coding: utf-8 -*-
"""
MJPEG Stream Reader Module
Handles connection to MJPEG streams, frame capturing, and stream management
"""
import logging
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests

from modules.constants import DEFAULT_IDLE_TIMEOUT, DEFAULT_WAKEUP_TIMEOUT
from modules.datatypes import MJPEGStreamError, ParameterValidator

logger = logging.getLogger(__name__)


class MJPEGStreamReader:
    """
    处理 MJPEG 流的读取，支持空闲超时和自动重连。
    """

    def __init__(
        self,
        mjpeg_url: str,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        wakeup_timeout: float = DEFAULT_WAKEUP_TIMEOUT,
    ):
        """
        初始化 MJPEG 流读取器。

        Args:
            mjpeg_url: MJPEG 流的 URL
            idle_timeout: 在关闭流连接前的不活动秒数
            wakeup_timeout: 等待流就绪的最大秒数

        Raises:
            MJPEGStreamError: 当初始化参数无效时
        """
        if not mjpeg_url:
            raise MJPEGStreamError("MJPEG URL 不能为空")

        try:
            # 验证 URL
            mjpeg_url = ParameterValidator.validate_mjpeg_url(mjpeg_url)
        except Exception as e:
            raise MJPEGStreamError(f"无效的 MJPEG URL: {e}")

        if idle_timeout <= 0:
            raise MJPEGStreamError("空闲超时必须大于 0")

        if wakeup_timeout <= 0:
            raise MJPEGStreamError("唤醒超时必须大于 0")

        self.mjpeg_url = mjpeg_url
        self.idle_timeout = idle_timeout
        self.wakeup_timeout = wakeup_timeout

        # 流状态
        self.stream = None
        self.active = False
        self.last_activity = 0

        # 当前帧状态
        self.current_frame = None
        self.frame_timestamp = 0
        self.frame_dimensions = (0, 0)  # width, height

        # 线程同步
        self.lock = threading.RLock()
        self.idle_monitor_thread = None
        self.reader_thread = None
        self.shutdown_event = threading.Event()

        logger.info(f"已初始化 MJPEG 流读取器，URL: {mjpeg_url}")
        logger.info(f"空闲超时: {idle_timeout}s, 唤醒超时: {wakeup_timeout}s")

    def start(self) -> bool:
        """
        启动 MJPEG 流读取器。

        Returns:
            bool: 如果成功启动则返回 True，否则返回 False

        Raises:
            MJPEGStreamError: 当启动过程中发生错误时
        """
        with self.lock:
            if self.active:
                logger.debug("流已经处于活动状态，忽略启动请求")
                return True

            try:
                logger.info("正在连接到 MJPEG 流...")
                self.stream = requests.get(self.mjpeg_url, stream=True, timeout=10)
                self.stream.raise_for_status()

                content_type = self.stream.headers.get("Content-Type", "")
                if "multipart/x-mixed-replace" not in content_type:
                    raise MJPEGStreamError(f"意外的内容类型: {content_type}")

                logger.info("已连接到 MJPEG 流")
                self.active = True
                self.last_activity = time.time()

                # 启动读取线程
                self.shutdown_event.clear()
                self.reader_thread = threading.Thread(
                    target=self._read_frames, daemon=True, name="MJPEGReaderThread"
                )
                self.reader_thread.start()

                # 启动空闲监视线程
                self.idle_monitor_thread = threading.Thread(
                    target=self._monitor_idle,
                    daemon=True,
                    name="MJPEGIdleMonitorThread",
                )
                self.idle_monitor_thread.start()

                return True

            except requests.exceptions.RequestException as e:
                error_msg = f"连接到 MJPEG 流失败: {e}"
                logger.error(error_msg)
                raise MJPEGStreamError(error_msg) from e
            except Exception as e:
                error_msg = f"启动 MJPEG 流读取器时出错: {e}"
                logger.error(error_msg)
                raise MJPEGStreamError(error_msg) from e
            finally:
                if not self.active and self.stream:
                    self.stream.close()
                    self.stream = None

    def stop(self) -> None:
        """
        停止 MJPEG 流读取器。

        Raises:
            MJPEGStreamError: 当停止过程中发生错误时
        """
        with self.lock:
            if not self.active:
                return

            logger.info("正在停止 MJPEG 流读取器...")
            self.shutdown_event.set()
            self.active = False

            try:
                if self.stream:
                    self.stream.close()
            except Exception as e:
                logger.warning(f"关闭流时出错: {e}")
            finally:
                self.stream = None

            # 等待线程结束
            try:
                if self.reader_thread and self.reader_thread.is_alive():
                    self.reader_thread.join(2.0)  # 等待最多 2 秒

                if self.idle_monitor_thread and self.idle_monitor_thread.is_alive():
                    self.idle_monitor_thread.join(2.0)  # 等待最多 2 秒
            except Exception as e:
                raise MJPEGStreamError(f"等待线程结束时出错: {e}")

            logger.info("MJPEG 流读取器已停止")

    def get_frame(self) -> Optional[Dict[str, Any]]:
        """
        获取流中的最新帧。

        Returns:
            包含帧数据的字典，如果没有可用帧则返回 None

        Raises:
            MJPEGStreamError: 当获取帧时发生错误
        """
        with self.lock:
            try:
                # 更新最后活动时间
                self.last_activity = time.time()

                # 如果流不活动，尝试启动
                if not self.active:
                    logger.info("流未激活，尝试启动")
                    success = self.start()

                    # 等待流就绪
                    start_time = time.time()
                    while success and self.current_frame is None:
                        if time.time() - start_time > self.wakeup_timeout:
                            raise MJPEGStreamError("等待流提供帧时超时")
                        time.sleep(0.1)  # 短暂睡眠以避免忙等待

                if self.current_frame is None:
                    return None

                return {
                    "frame": self.current_frame.copy(),
                    "timestamp": self.frame_timestamp,
                    "dimensions": self.frame_dimensions,
                }

            except MJPEGStreamError:
                raise
            except Exception as e:
                raise MJPEGStreamError(f"获取帧时出错: {e}")

    def _read_frames(self) -> None:
        """
        在后台线程中读取 MJPEG 流的帧。

        Raises:
            MJPEGStreamError: 当读取帧时发生错误
        """
        if not self.stream:
            error_msg = "无法读取帧：stream 未初始化"
            logger.error(error_msg)
            with self.lock:
                self.active = False
            raise MJPEGStreamError(error_msg)

        bytes_buffer = bytes()
        consecutive_decode_errors = 0
        MAX_CONSECUTIVE_DECODE_ERRORS = 10

        consecutive_empty_chunks = 0
        MAX_CONSECUTIVE_EMPTY_CHUNKS = 5

        try:
            for chunk in self.stream.iter_content(chunk_size=1024):
                if self.shutdown_event.is_set():
                    logger.debug("读取线程检测到关闭事件")
                    break

                if not chunk:
                    consecutive_empty_chunks += 1
                    if consecutive_empty_chunks > MAX_CONSECUTIVE_EMPTY_CHUNKS:
                        logger.error(
                            f"连续 {MAX_CONSECUTIVE_EMPTY_CHUNKS} 次接收到空数据块，停止读取。"
                        )
                        raise MJPEGStreamError("连续多次接收到空数据块")
                    continue

                consecutive_empty_chunks = 0
                bytes_buffer += chunk

                # 查找 JPEG 开始和结束标记
                jpg_start = bytes_buffer.find(b"\xff\xd8")
                if jpg_start == -1:
                    continue

                jpg_end = bytes_buffer.find(b"\xff\xd9", jpg_start)
                if jpg_end == -1:
                    continue

                # 提取和解码 JPEG 数据
                jpg_data = bytes_buffer[jpg_start : jpg_end + 2]
                bytes_buffer = bytes_buffer[jpg_end + 2 :]

                try:
                    # 解码图像
                    frame_array = np.frombuffer(jpg_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    if frame is None:
                        logger.warning("无法解码帧 (cv2.imdecode 返回 None)")
                        consecutive_decode_errors += 1
                        if consecutive_decode_errors > MAX_CONSECUTIVE_DECODE_ERRORS:
                            logger.error(
                                f"连续 {MAX_CONSECUTIVE_DECODE_ERRORS} 次解码帧失败，停止读取。"
                            )
                            raise MJPEGStreamError("连续解码帧失败")
                        continue

                    consecutive_decode_errors = 0

                    # 更新帧数据
                    with self.lock:
                        self.current_frame = frame
                        self.frame_timestamp = time.time()
                        self.frame_dimensions = (frame.shape[1], frame.shape[0])

                except Exception as e:
                    logger.error(f"处理帧数据时出错: {e}")
                    continue

        except MJPEGStreamError:
            raise
        except Exception as e:
            raise MJPEGStreamError(f"读取帧时出错: {e}")
        finally:
            with self.lock:
                self.active = False
                if self.stream:
                    try:
                        self.stream.close()
                    except Exception as e:
                        logger.error(f"关闭流时出错: {e}")
                    self.stream = None

    def _monitor_idle(self) -> None:
        """
        监视流的空闲状态并在必要时关闭连接。
        """
        while not self.shutdown_event.is_set():
            try:
                with self.lock:
                    if self.active:
                        idle_time = time.time() - self.last_activity
                        if idle_time > self.idle_timeout:
                            logger.info(
                                f"流空闲超过 {self.idle_timeout} 秒，正在关闭连接"
                            )
                            self.stop()

                time.sleep(1)  # 检查间隔

            except Exception as e:
                logger.error(f"监视空闲状态时出错: {e}")
                # 继续监视，不要因为一个错误就停止

    def health_check(self) -> Dict[str, Any]:
        """
        检查流的健康状态。

        Returns:
            包含健康状态信息的字典

        Raises:
            MJPEGStreamError: 当检查健康状态时发生错误
        """
        try:
            with self.lock:
                current_time = time.time()
                # 使用一个大的但有限的数字代替 inf
                frame_age = (
                    current_time - self.frame_timestamp
                    if self.frame_timestamp > 0
                    else 1e6  # 使用 1000000 秒（约 11.6 天）作为最大值
                )

                return {
                    "active": self.active,
                    "frame_valid": self.current_frame is not None,
                    "frame_age_seconds": frame_age,
                    "frame_resolution": f"{self.frame_dimensions[0]}x{self.frame_dimensions[1]}",
                    "last_activity": self.last_activity,
                    "idle_time": current_time - self.last_activity,
                }

        except Exception as e:
            raise MJPEGStreamError(f"检查健康状态时出错: {e}")

    def update_mjpeg_url(self, new_url: str) -> bool:
        """
        更新 MJPEG 流的 URL。

        Args:
            new_url: 新的 MJPEG URL

        Returns:
            如果更新成功则返回 True，否则返回 False

        Raises:
            MJPEGStreamError: 当更新 URL 时发生错误
        """
        if not new_url:
            raise MJPEGStreamError("新的 URL 不能为空")

        try:
            # 验证新 URL
            new_url = ParameterValidator.validate_mjpeg_url(new_url)

            # 停止当前流
            self.stop()

            # 更新 URL
            self.mjpeg_url = new_url

            # 尝试启动新流
            return self.start()

        except Exception as e:
            raise MJPEGStreamError(f"更新 MJPEG URL 时出错: {e}")

    def close(self) -> None:
        """
        关闭 MJPEG 流读取器。

        Raises:
            MJPEGStreamError: 当关闭时发生错误
        """
        try:
            self.stop()
        except Exception as e:
            raise MJPEGStreamError(f"关闭 MJPEG 流读取器时出错: {e}")

    def __del__(self):
        """
        析构函数，确保资源被正确释放。
        """
        try:
            self.stop()
        except Exception as e:
            logger.error(f"在析构函数中停止 MJPEG 流读取器时出错: {e}")
