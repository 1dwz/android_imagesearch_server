# -*- coding: utf-8 -*-
"""
MJPEG stream reader with enhanced resource management and error handling.
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import cv2
import requests

# Import FrameDataCache from state_managers
from .state_managers import FrameDataCache  # Add this import
from .utils import save_raw_debug_frame

logger = logging.getLogger(__name__)

# Configuration constants
MJPEG_CONNECT_RETRY_WAIT = 1.0
MJPEG_POST_CONNECT_WAIT = 2.0
MJPEG_READ_FAIL_BASE_DELAY = 1.0
MJPEG_MAX_READ_FAILS = 5
MJPEG_RECONNECT_DELAY = 3.0
SHUTDOWN_CONFIRM_TIMEOUT = 5.0
SHUTDOWN_CONFIRM_INTERVAL = 0.2
THREAD_SHUTDOWN_TIMEOUT_LONG = 60.0
THREAD_SHUTDOWN_TIMEOUT_SHORT = 5.0


def _release_capture(cap: Any, context: str) -> None:
    """Safely release video capture object"""
    if cap is not None:
        try:
            cap.release()
            logger.debug(f"[MJPEG Thread] Released capture object ({context})")
        except Exception as e:
            logger.warning(
                f"[MJPEG Thread] Error releasing capture ({context}): {str(e)}"
            )


def _configure_logging(level: str) -> None:
    """Configures logging for the MJPEG reader."""
    # This reader might run in a separate thread, so configure its logger directly
    # if it's not being configured by the main application's basicConfig
    # In this project, logging is configured in start_fastapi.py, so this might not be strictly needed
    # but keeping it as a safeguard if this module were used standalone.
    # logger.setLevel(level.upper())
    pass  # Logging is configured externally


class MJPEGStreamManager:
    """
    Manages the MJPEG stream connection and frame reading in a separate thread.
    """

    def __init__(
        self,
        mjpeg_url: str,
        frame_cache: FrameDataCache,
        debug_executor: Optional[ThreadPoolExecutor],
        debug_save_dir: str,
        mjpeg_ready_event: threading.Event,
        idle_timeout_seconds: float,
        wakeup_timeout_seconds: float,
    ):
        self._mjpeg_url = mjpeg_url
        self._frame_cache = frame_cache
        self._debug_executor = debug_executor
        self._debug_save_dir = debug_save_dir
        self._stream_ready_event = mjpeg_ready_event
        self._idle_timeout_seconds = idle_timeout_seconds
        self._wakeup_timeout = wakeup_timeout_seconds
        self._idle_monitor_thread = None
        self._monitor_stop_event = threading.Event()
        self._stream_stop_event = threading.Event()
        self._thread_shutdown_timeout_short = THREAD_SHUTDOWN_TIMEOUT_SHORT
        self._thread_shutdown_timeout_long = THREAD_SHUTDOWN_TIMEOUT_LONG
        self._stream_thread = None
        logger.debug(
            f"MJPEGStreamManager initialized with url={mjpeg_url}, idle_timeout={idle_timeout_seconds}, wakeup_timeout={wakeup_timeout_seconds}"
        )

    def _read_stream_loop(self) -> None:
        """
        Internal loop to connect to and read frames from the MJPEG stream.
        Handles connection retries and frame read errors.
        This is the core stream reading logic extracted from the old _run method.
        """
        cap = None
        read_fail_count = 0

        try:
            while not self._stream_stop_event.is_set():
                try:
                    # Connection phase
                    if cap is None:
                        self._frame_cache.update_status("Connecting...")
                        self._stream_ready_event.clear()

                        logger.debug(
                            "MJPEG Thread: Connecting to stream: %s", self._mjpeg_url
                        )
                        try:
                            cap = cv2.VideoCapture(self._mjpeg_url)
                            time.sleep(MJPEG_CONNECT_RETRY_WAIT)

                            if not cap.isOpened():
                                logger.warning(
                                    "MJPEG Thread: Connection attempt failed (capture not opened)"
                                )
                                cap = None
                                read_fail_count += 1

                                if read_fail_count > MJPEG_MAX_READ_FAILS:
                                    self._frame_cache.update_status(
                                        "Stream Failed (Max Errors)"
                                    )
                                    logger.error(
                                        "MJPEG Thread: Connection failed too many times"
                                    )
                                    break  # Exit the while loop on too many connection failures

                                time.sleep(MJPEG_RECONNECT_DELAY)
                                continue

                            logger.debug(
                                "MJPEG Thread: Connected to stream successfully"
                            )
                            read_fail_count = 0
                            time.sleep(MJPEG_POST_CONNECT_WAIT)

                        except Exception:
                            logger.exception("MJPEG Thread: Connection error:")
                            self._frame_cache.update_status("Connection Failed")
                            _release_capture(cap, "connect")
                            cap = None
                            time.sleep(MJPEG_RECONNECT_DELAY)
                            continue

                    # Read frame phase
                    try:
                        frame_timestamp = time.time()
                        success, frame = cap.read()

                        if not success or frame is None or frame.size == 0:
                            logger.warning("MJPEG Thread: Failed to read frame")
                            read_fail_count += 1

                            if read_fail_count > MJPEG_MAX_READ_FAILS:
                                self._frame_cache.update_status(
                                    "Stream Failed (Max Errors)"
                                )
                                logger.error(
                                    "MJPEG Thread: Frame read failed too many times"
                                )
                                _release_capture(cap, "read")
                                cap = None
                                read_fail_count = 0
                                continue  # Restart connection attempt

                            time.sleep(
                                MJPEG_READ_FAIL_BASE_DELAY * min(read_fail_count, 5)
                            )
                            continue

                        # Update cache with new frame
                        self._frame_cache.update_frame(frame, frame_timestamp)
                        read_fail_count = 0

                        # Signal that the stream is ready after the first successful frame read
                        if not self._stream_ready_event.is_set():
                            self._stream_ready_event.set()
                            logger.debug(
                                "MJPEG Thread: First frame read successfully. Signaling stream readiness."
                            )

                        # Handle debug image saving
                        if self._debug_executor and self._debug_save_dir:
                            if hasattr(
                                self._stream_ready_event, "max_debug_files"
                            ):  # Check if stream_ready_event has max_debug_files
                                try:
                                    self._debug_executor.submit(
                                        save_raw_debug_frame,
                                        frame.copy(),
                                        frame_timestamp,
                                        self._debug_save_dir,
                                        getattr(
                                            self._stream_ready_event,
                                            "max_debug_files",
                                            100,
                                        ),
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "Failed to submit debug image task: %s", str(e)
                                    )

                    except cv2.error as e:
                        logger.exception("OpenCV error reading frame:")
                        self._frame_cache.update_status("OpenCV Error")
                        _release_capture(cap, "opencv")
                        cap = None
                        time.sleep(MJPEG_RECONNECT_DELAY)

                    except Exception:
                        logger.exception("Unexpected error reading frame:")
                        self._frame_cache.update_status("Unexpected Error")
                        _release_capture(cap, "unexpected")
                        cap = None
                        time.sleep(MJPEG_RECONNECT_DELAY)

                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"MJPEG stream connection error: {e}. Retrying in {MJPEG_CONNECT_RETRY_WAIT}s."
                    )
                    logger.debug(f"Caught RequestException: {e}. Retrying connection.")
                    if self._stream_ready_event.is_set():
                        self._stream_ready_event.clear()  # Stream is not ready on connection error
                    logger.debug("Stream ready event cleared due to connection error.")
                    time.sleep(MJPEG_CONNECT_RETRY_WAIT)
                except Exception as e:
                    logger.exception("Unexpected error in MJPEG stream thread:")
                    logger.debug(
                        f"Caught unexpected exception in _read_stream_loop: {e}"
                    )
                    if self._stream_ready_event.is_set():
                        self._stream_ready_event.clear()  # Stream is not ready on unexpected error
                    logger.debug("Stream ready event cleared due to unexpected error.")
                    # Sleep for a bit before retrying connection to avoid tight loop
                    time.sleep(MJPEG_RECONNECT_DELAY)

                finally:
                    if cap is not None:
                        logger.debug("Releasing video capture object.")
                        cap.release()
                        cap = None  # Ensure cap is None after release
                        logger.debug("Video capture object released.")

            logger.info("MJPEG stream thread stopping due to stop event.")
            logger.debug("_read_stream_loop finished. Stream stop event was set.")

        except Exception:
            logger.exception("Critical error in MJPEG stream runner thread:")
            logger.debug("Caught critical exception in _run thread.")
        finally:
            if cap is not None:
                logger.debug("Releasing video capture object in final cleanup.")
                cap.release()
                logger.debug("Video capture object released in final cleanup.")
            if self._stream_ready_event.is_set():
                self._stream_ready_event.clear()  # Ensure ready event is cleared on exit
            logger.debug("Stream ready event cleared on thread exit.")
            logger.info("MJPEG stream thread finished.")
            logger.debug("_run method completely finished.")

    def start(self) -> None:
        if self._stream_thread is None or not self._stream_thread.is_alive():
            logger.info("Starting MJPEG stream thread...")
            self._stream_stop_event.clear()  # Ensure stop event is clear before starting
            self._stream_ready_event.clear()  # Ensure ready event is clear before starting
            self._stream_thread = threading.Thread(
                target=self._read_stream_loop, name="MJPEGStreamThread"
            )
            self._stream_thread.daemon = (
                True  # Allow main program to exit even if this thread is running
            )
            self._stream_thread.start()
            logger.debug(
                f"MJPEG stream thread started with name: {self._stream_thread.name}"
            )

        if (
            self._idle_monitor_thread is None
            or not self._idle_monitor_thread.is_alive()
        ):
            logger.info("Starting MJPEG idle monitor thread...")
            self._monitor_stop_event.clear()
            self._idle_monitor_thread = threading.Thread(
                target=self._idle_monitor, name="MJPEGIdleMonitorThread"
            )
            self._idle_monitor_thread.daemon = True
            self._idle_monitor_thread.start()
            logger.debug(
                f"MJPEG idle monitor thread started with name: {self._idle_monitor_thread.name}"
            )

    def stop(self, timeout: float = THREAD_SHUTDOWN_TIMEOUT_LONG) -> None:
        logger.info("Stopping MJPEG stream and idle monitor threads...")
        # Signal threads to stop
        self._stream_stop_event.set()
        logger.debug("Stream stop event set.")
        self._monitor_stop_event.set()
        logger.debug("Monitor stop event set.")

        threads_to_join = []
        if self._stream_thread and self._stream_thread.is_alive():
            threads_to_join.append(("MJPEG stream thread", self._stream_thread))
            logger.debug("Added stream thread to join list.")
        if self._idle_monitor_thread and self._idle_monitor_thread.is_alive():
            threads_to_join.append(
                ("MJPEG idle monitor thread", self._idle_monitor_thread)
            )
            logger.debug("Added monitor thread to join list.")

        # Try joining threads with timeout
        join_timeout = timeout / len(threads_to_join) if threads_to_join else timeout
        logger.debug(
            f"Attempting to join threads with individual timeout: {join_timeout:.2f}s"
        )
        for thread_name, thread in threads_to_join:
            try:
                logger.debug(f"Joining thread: {thread_name}...")
                thread.join(join_timeout)
                if thread.is_alive():
                    logger.warning(
                        f"Thread {thread_name} did not finish within timeout."
                    )
                    logger.debug(
                        f"Thread {thread_name} is still alive after join timeout."
                    )
                else:
                    logger.debug(f"Thread {thread_name} joined successfully.")
            except Exception as e:
                logger.error(f"Error joining thread {thread_name}: {e}")
                logger.debug(
                    f"Caught exception while joining thread {thread_name}: {e}"
                )

        self._stream_thread = None
        self._idle_monitor_thread = None
        logger.debug("Thread objects set to None.")
        logger.info("MJPEG stream and idle monitor threads stopped.")
        logger.debug("stop method finished.")

    def start_if_needed(self) -> bool:
        # Check if the stream thread is not alive
        if self._stream_thread is None or not self._stream_thread.is_alive():
            logger.info("MJPEG stream is not running. Checking idle status...")
            # Check if the frame cache is currently idle
            frame_data, frame_timestamp, frame_status = (
                self._frame_cache.get_frame_data()
            )
            if self._frame_cache.is_idle(self._idle_timeout_seconds):
                logger.info(
                    f"Frame cache is idle (>{self._idle_timeout_seconds:.2f}s). Starting stream."
                )
                self.start()
                # Set a flag to indicate that we should wait for the stream to become ready
                should_wait = True
                logger.debug("Set should_wait flag to True.")
            else:
                logger.debug("Frame cache is not idle. No need to start stream.")
                should_wait = False

            # After releasing the lock, start the stream if needed
            if should_wait:
                logger.debug(
                    "Should wait is True. Attempting to wait for stream to be ready."
                )
                # Start the stream if not already started
                # This call to start() might be redundant if self.start() was called above, but is safe
                # It ensures the thread is definitely started before waiting
                if self._stream_thread is None or not self._stream_thread.is_alive():
                    logger.debug("Stream thread is not alive. Starting stream.")
                    self.start()
                else:
                    logger.debug("Stream thread is alive. Waiting for ready event.")

                # Wait for the stream to become ready
                ready_wait_start = time.monotonic()
                is_ready = self._stream_ready_event.wait(self._wakeup_timeout)
                wait_duration = time.monotonic() - ready_wait_start

                if self._stream_stop_event.is_set():
                    logger.warning(
                        f"start_if_needed: Stream was stopped during {wait_duration:.2f}s wait"
                    )
                    logger.debug("Stream stopped during wait. Returning False.")
                    return False

                if not is_ready:
                    logger.warning(
                        f"start_if_needed: Wait for ready timeout after {wait_duration:.2f}s."
                    )
                    logger.debug("Wait for ready timed out. Returning False.")
                    return False

                logger.info(
                    f"start_if_needed: Stream became ready after {wait_duration:.2f}s wait"
                )
                logger.debug(
                    "Stream ready after waiting for ready event. Returning True."
                )
                return True

        else:
            # Stream thread is already alive
            logger.debug("MJPEG stream thread is already running.")
            if not self._stream_ready_event.is_set():
                logger.debug(
                    "Stream thread is alive but ready event is not set. Waiting for ready..."
                )
                # If the thread is alive but not ready, wait for it to become ready
                ready_wait_start = time.monotonic()
                is_ready = self._stream_ready_event.wait(self._wakeup_timeout)
                wait_duration = time.monotonic() - ready_wait_start

                if self._stream_stop_event.is_set():
                    logger.warning(
                        f"start_if_needed: Stream was stopped during wait for ready ({wait_duration:.2f}s)"
                    )
                    logger.debug(
                        "Stream stopped during wait for ready. Returning False."
                    )
                    return False

                if not is_ready:
                    logger.warning(
                        f"start_if_needed: Wait for ready timeout after {wait_duration:.2f}s."
                    )
                    logger.debug("Wait for ready timed out. Returning False.")
                    return False

                logger.info(
                    f"start_if_needed: Stream became ready after {wait_duration:.2f}s wait"
                )
                logger.debug(
                    "Stream ready after waiting for ready event. Returning True."
                )
                return True
            else:
                # Thread is alive and ready
                logger.debug("MJPEG stream thread is alive and ready. Returning True.")
                return True

    def wait_for_stream_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Waits for the MJPEG stream to become ready (i.e., the first frame is successfully read).

        Args:
            timeout: The maximum number of seconds to wait. If None, it will block indefinitely.

        Returns:
            True if the stream became ready within the timeout, False otherwise.
        """
        logger.debug(f"Waiting for stream ready event with timeout: {timeout}")
        is_ready = self._stream_ready_event.wait(timeout=timeout)
        if is_ready:
            logger.debug("Stream ready event received.")
        else:
            logger.warning("Timeout waiting for stream ready event.")
        return is_ready

    def get_status(self) -> Dict[str, Any]:
        """
        Get stream status based on thread state and ready event
        """
        # Get stream status based on thread state and ready event
        is_stream_thread_alive = (
            self._stream_thread is not None and self._stream_thread.is_alive()
        )
        is_stream_ready = self._stream_ready_event.is_set()
        logger.debug(
            f"Getting status: stream_thread_alive={is_stream_thread_alive}, stream_ready={is_stream_ready}"
        )

        if not is_stream_thread_alive:
            logger.debug("Status: Stream thread not alive. Returning status 'Stopped'.")
            return {"status": "Stopped", "message": "Stream thread not running"}

        if not is_stream_ready:
            logger.debug(
                "Status: Stream thread alive but not ready. Returning status 'Connecting/Buffering'."
            )
            return {
                "status": "Connecting/Buffering",
                "message": "Stream thread running but not ready",
            }

        # If thread is alive and ready, check frame cache status
        frame_data, frame_timestamp, frame_status = self._frame_cache.get_frame_data()
        frame_age = (
            time.monotonic() - frame_timestamp if frame_timestamp else 99999999.0
        )
        if frame_data.get("status") == "Idle":
            message = f"Idle (Last Frame @ {frame_data.get('timestamp_str')}, Age: {frame_age:.1f}s)"
            logger.debug(
                f"Status: Stream ready and frame cache is idle. Message: {message}"
            )
            return {
                "status": "Idle",
                "message": message,
                "frame_age": frame_age,
                "timestamp_str": frame_data.get("timestamp_str"),
            }
        message = f"Running (Frame @ {frame_data.get('timestamp_str')}), Age: {frame_age:.1f}s"
        logger.debug(
            f"Status: Stream ready and frame cache is active. Message: {message}"
        )
        return {
            "status": "Running",
            "message": message,
            "frame_age": frame_age,
            "timestamp_str": frame_data.get("timestamp_str"),
        }

    def _idle_monitor(self) -> None:
        """
        Monitors frame activity and stops the stream if idle for too long.
        """
        logger.debug("Idle monitor thread started.")
        while not self._monitor_stop_event.is_set():
            # Check idle status periodically
            self._monitor_stop_event.wait(1.0)  # Check every second
            if self._monitor_stop_event.is_set():
                logger.debug("Monitor stop event set. Exiting idle monitor.")
                break

            # Only check idle if stream is supposed to be active (thread is running and not stopping)
            if (
                self._stream_thread
                and self._stream_thread.is_alive()
                and not self._stream_stop_event.is_set()
            ):
                logger.debug(
                    "Idle monitor: Stream thread alive and not stopping. Checking idle status..."
                )
                if self._idle_timeout_seconds != float(
                    "inf"
                ) and self._frame_cache.is_idle(self._idle_timeout_seconds):
                    logger.info(
                        f"MJPEG stream idle for >{self._idle_timeout_seconds:.2f}s. Stopping stream."
                    )
                    logger.debug("Idle monitor: Frame cache is idle. Stopping stream.")
                    self.stop(
                        timeout=THREAD_SHUTDOWN_TIMEOUT_SHORT
                    )  # Use short timeout for stop during idle
                else:
                    logger.debug(
                        "Idle monitor: Stream is not idle or idle timeout is infinite."
                    )
            else:
                logger.debug(
                    "Idle monitor: Stream thread not active or is stopping. Skipping idle check."
                )

        logger.info("MJPEG idle monitor thread finished.")
        logger.debug("_idle_monitor method completely finished.")

    def reset_idle_timer(self) -> None:
        """
        重置帧缓存的空闲计时器，表示活动发生。
        这会防止空闲监视器认为流没有被使用而停止它。
        """
        logger.debug("Resetting frame cache idle timer")
        self._frame_cache.reset_idle_timer()
        logger.debug("Frame cache idle timer reset")
