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

from modules.constants import (
    DEFAULT_IDLE_TIMEOUT,  # Ensure these are defined in constants
)
from modules.constants import (
    DEFAULT_WAKEUP_TIMEOUT,  # Ensure these are defined in constants
)
from modules.constants import (
    MAX_CONSECUTIVE_DECODE_ERRORS,
    MAX_CONSECUTIVE_EMPTY_CHUNKS,
    MAX_MJPEG_BUFFER_SIZE,
    MJPEG_BUFFER_KEEP_SIZE_ON_TRIM,
)
from modules.datatypes import MJPEGStreamError, ParameterValidator, ValidationError
from modules.state_managers import FrameDataCache  # For type hinting or direct use

logger = logging.getLogger(__name__)


class MJPEGStreamReader:
    """
    Manages reading an MJPEG stream with support for idle timeout, auto-reconnection,
    and dynamic URL updates with rollback capability.
    """

    def __init__(
        self,
        mjpeg_url: str,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        wakeup_timeout: float = DEFAULT_WAKEUP_TIMEOUT,
        frame_cache: Optional[FrameDataCache] = None,  # Optional: Inject FrameDataCache
    ):
        try:
            self.mjpeg_url = ParameterValidator.validate_mjpeg_url(mjpeg_url)
        except ValidationError as e:  # Catch specific validation error
            raise MJPEGStreamError(
                f"Initial MJPEG URL '{mjpeg_url}' is invalid: {e}"
            ) from e

        if not isinstance(idle_timeout, (int, float)) or idle_timeout <= 0:
            raise MJPEGStreamError(
                f"Idle timeout must be a positive number, got {idle_timeout}"
            )
        if not isinstance(wakeup_timeout, (int, float)) or wakeup_timeout <= 0:
            raise MJPEGStreamError(
                f"Wakeup timeout must be a positive number, got {wakeup_timeout}"
            )

        self.idle_timeout = idle_timeout
        self.wakeup_timeout = wakeup_timeout
        self.frame_cache = (
            frame_cache if frame_cache else FrameDataCache()
        )  # Use provided or create one

        self._active_url_before_update: Optional[str] = (
            None  # Stores the URL that was active before an update attempt
        )
        self._request: Optional[requests.Response] = None
        self._stream: Optional[Any] = None  # Iterator from request.iter_content
        self.active = False  # Is the stream currently connected and reading?

        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        self.reader_thread: Optional[threading.Thread] = None
        self.idle_monitor_thread: Optional[threading.Thread] = None

        # Statistics/status
        self.consecutive_decode_errors = 0
        self.consecutive_empty_chunks = 0
        self.total_frames_decoded = 0

        logger.info(
            f"MJPEGStreamReader initialized for URL: {self.mjpeg_url} (Idle: {idle_timeout}s, Wakeup: {wakeup_timeout}s)"
        )

    def _get_thread_name_suffix(self) -> str:
        """Generates a suffix for thread names based on the current MJPEG URL."""
        try:
            # Extract a hostname-like part or a significant part of the path
            url_part = self.mjpeg_url.split("//")[-1].split("/")[0].split("?")[0]
            return f"-{url_part[:30]}"  # Limit length
        except Exception:
            return "-UnknownURL"

    def start(self) -> bool:
        """
        Starts the MJPEG stream reader thread and idle monitor.
        Returns True if successfully started, False otherwise.
        Sets self._active_url_before_update on successful start.
        """
        with self.lock:
            if self.active:
                logger.debug(f"Stream for {self.mjpeg_url} is already active.")
                return True

            try:
                logger.info(f"Attempting to connect to MJPEG stream: {self.mjpeg_url}")
                # Set a reasonable timeout for the initial GET request
                self._request = requests.get(
                    self.mjpeg_url, stream=True, timeout=(5.0, 10.0)
                )  # connect, read
                self._request.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

                content_type = self._request.headers.get("Content-Type", "").lower()
                if "multipart/x-mixed-replace" not in content_type:
                    if self._request:
                        self._request.close()
                    self._request = None
                    logger.error(
                        f"Unexpected content type for {self.mjpeg_url}: {content_type}"
                    )
                    return False  # Not an MJPEG stream

                self._stream = self._request.iter_content(
                    chunk_size=8192
                )  # Adjusted chunk size

                self.active = True  # Mark as active *before* starting threads
                self.last_activity_time = time.time()  # For idle monitor
                self._active_url_before_update = (
                    self.mjpeg_url
                )  # This URL is now considered active
                self.frame_cache.update_status("Connecting", self.last_activity_time)

                self.shutdown_event.clear()  # Ensure event is clear before starting threads

                thread_suffix = self._get_thread_name_suffix()
                self.reader_thread = threading.Thread(
                    target=self._read_frames_loop,
                    daemon=True,
                    name=f"MJPEGReader{thread_suffix}",
                )
                self.reader_thread.start()

                if (
                    self.idle_monitor_thread is None
                    or not self.idle_monitor_thread.is_alive()
                ):
                    self.idle_monitor_thread = threading.Thread(
                        target=self._monitor_idle_loop,
                        daemon=True,
                        name=f"MJPEGIdleMon{thread_suffix}",
                    )
                    self.idle_monitor_thread.start()

                logger.info(
                    f"Successfully connected and started reader for MJPEG stream: {self.mjpeg_url}"
                )
                return True

            except requests.exceptions.Timeout:
                logger.error(f"Timeout connecting to MJPEG stream {self.mjpeg_url}.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to MJPEG stream {self.mjpeg_url}: {e}")
            except Exception as e:  # Catch any other unexpected error during start
                logger.exception(
                    f"Unexpected error starting MJPEG stream reader for {self.mjpeg_url}: {e}"
                )

            # If any exception occurred, ensure cleanup
            self.active = False
            if self._request:
                self._request.close()
            self._request = None
            self._stream = None
            self.frame_cache.update_status(
                f"Connection Failed: {self.mjpeg_url}", time.time()
            )
            return False

    def stop(self) -> None:
        """Stops the MJPEG stream reader, closing connections and joining threads."""
        with self.lock:
            if not self.active and not (
                self.reader_thread and self.reader_thread.is_alive()
            ):
                logger.debug(
                    f"Stream {self.mjpeg_url} already stopped or reader thread not running."
                )
                # return # Allow stop to proceed to ensure shutdown_event is set for idle monitor

            logger.info(f"Stopping MJPEG stream reader for {self.mjpeg_url}...")
            self.shutdown_event.set()

            # Set active to False immediately to prevent get_frame from trying to restart
            # and to signal _read_frames_loop to stop if it's in a retry sleep.
            was_active = self.active
            self.active = False

            # Close the network connection
            current_request = self._request
            self._request = None  # Nullify before closing to prevent race
            self._stream = None
            if current_request:
                try:
                    current_request.close()
                    logger.debug(f"Closed requests connection for {self.mjpeg_url}.")
                except Exception as e:
                    logger.warning(
                        f"Error closing requests connection for {self.mjpeg_url}: {e}"
                    )

            # Join reader thread
            current_reader_thread = self.reader_thread
            if current_reader_thread and current_reader_thread.is_alive():
                logger.debug(
                    f"Waiting for reader thread of {self.mjpeg_url} to join..."
                )
                current_reader_thread.join(
                    timeout=self.wakeup_timeout + 1.0
                )  # Give it enough time
                if current_reader_thread.is_alive():
                    logger.warning(
                        f"Reader thread for {self.mjpeg_url} did not join in time."
                    )
            self.reader_thread = None

            # Idle monitor thread can be left running if it's designed to persist,
            # or stopped here if it's per-stream-session. Let's assume it stops.
            current_idle_monitor_thread = self.idle_monitor_thread
            if current_idle_monitor_thread and current_idle_monitor_thread.is_alive():
                logger.debug(
                    f"Waiting for idle monitor thread of {self.mjpeg_url} to join..."
                )
                current_idle_monitor_thread.join(timeout=1.0)
            # self.idle_monitor_thread = None # Decide if it should be cleared

            if was_active:  # Only update status if it was previously active
                self.frame_cache.update_status("Stopped", time.time())
            logger.info(f"MJPEG stream reader for {self.mjpeg_url} stopped.")

    def get_frame(self) -> Optional[Dict[str, Any]]:
        """
        Gets the latest frame data from the internal FrameDataCache.
        Initiates stream start if not active.
        """
        self.last_activity_time = time.time()  # Record activity even on get_frame call
        self.frame_cache.reset_idle_timer()

        with self.lock:  # Protect self.active and self.start() call
            if not self.active:
                logger.info(
                    f"Stream {self.mjpeg_url} is not active. Attempting to start..."
                )
                if not self.start():  # start() is internally locked
                    logger.warning(
                        f"Failed to start stream {self.mjpeg_url} on demand from get_frame."
                    )
                    self.frame_cache.update_status(
                        "Start Failed on Demand", time.time()
                    )
                    return None  # Start failed

                # Wait for the first frame after starting
                wait_start_time = time.time()
                while (
                    self.frame_cache.get_data().get("timestamp", 0.0) == 0.0
                    and (time.time() - wait_start_time) < self.wakeup_timeout
                ):
                    if (
                        self.shutdown_event.is_set() or not self.active
                    ):  # Stream might have stopped again
                        logger.warning(
                            f"Stream {self.mjpeg_url} stopped while waiting for first frame."
                        )
                        return None
                    time.sleep(0.1)

                if self.frame_cache.get_data().get("timestamp", 0.0) == 0.0:
                    logger.warning(
                        f"Timeout waiting for the first frame from {self.mjpeg_url} after on-demand start."
                    )
                    self.frame_cache.update_status(
                        "No First Frame Timeout", time.time()
                    )
                    return None

        # At this point, stream should be active (or start was attempted)
        # FrameDataCache provides the actual frame data and its own locking
        frame_data = self.frame_cache.get_data()
        if frame_data.get("frame_copy") is not None:
            return {  # Construct a dictionary similar to old get_frame structure if needed by callers
                "frame": frame_data[
                    "frame_copy"
                ],  # Assuming frame_copy is the color frame
                "timestamp": frame_data["timestamp"],
                "dimensions": (
                    frame_data["shape"][:2] if frame_data["shape"] else (0, 0)
                ),  # width, height
            }
        return None

    def _read_frames_loop(self) -> None:
        """
        Main loop for the reader thread. Handles reading, decoding, and reconnection.
        """
        logger.info(f"Reader thread started for {self.mjpeg_url}.")
        byte_buffer = bytearray()

        retry_delay = 1.0  # Initial retry delay in seconds
        max_retry_delay = 30.0  # Max retry delay

        while not self.shutdown_event.is_set():
            if not self.active or not self._stream or not self._request:
                logger.info(
                    f"Reader loop: Stream {self.mjpeg_url} not active or connection objects missing. Attempting to re-establish."
                )
                self.frame_cache.update_status("Reconnecting", time.time())

                with self.lock:  # Acquire lock before calling start() if it also locks
                    if self.shutdown_event.is_set():
                        break  # Check again after acquiring lock
                    if self._request:  # Close previous request if any
                        try:
                            self._request.close()
                        except:
                            pass
                    self._request = None
                    self._stream = None
                    self.active = False  # Mark inactive before attempting start

                    if not self.start():  # start() will try to connect
                        logger.error(
                            f"Reader loop: Failed to re-establish stream {self.mjpeg_url}. Retrying in {retry_delay:.1f}s."
                        )
                        self.shutdown_event.wait(retry_delay)  # Wait or until shutdown
                        retry_delay = min(
                            retry_delay * 1.5, max_retry_delay
                        )  # Exponential backoff
                        continue  # Retry connection
                    else:
                        logger.info(
                            f"Reader loop: Successfully re-established stream {self.mjpeg_url}."
                        )
                        retry_delay = 1.0  # Reset retry delay on success
                        byte_buffer.clear()  # Clear buffer on new stream
                        self.consecutive_decode_errors = 0
                        self.consecutive_empty_chunks = 0

            try:
                # Ensure _stream is not None (should be set by successful start)
                if not self._stream:
                    logger.warning(
                        f"Reader loop: _stream is None for {self.mjpeg_url} despite active=True. Resetting."
                    )
                    self.active = False  # Force re-check and re-start
                    continue

                chunk = next(self._stream, None)  # Get next chunk
                self.last_activity_time = (
                    time.time()
                )  # Update activity time on receiving data/chunk

                if self.shutdown_event.is_set():
                    break

                if chunk is None:  # Stream ended from server side
                    logger.warning(
                        f"Stream {self.mjpeg_url} ended (received None chunk). Will attempt to reconnect."
                    )
                    self.active = False  # Trigger reconnect
                    self.frame_cache.update_status(
                        "Stream Ended by Server", time.time()
                    )
                    continue

                if not chunk:  # Empty chunk
                    self.consecutive_empty_chunks += 1
                    if self.consecutive_empty_chunks >= MAX_CONSECUTIVE_EMPTY_CHUNKS:
                        logger.warning(
                            f"Stream {self.mjpeg_url}: Received {self.consecutive_empty_chunks} empty chunks. Reconnecting."
                        )
                        self.active = False  # Trigger reconnect
                        self.frame_cache.update_status(
                            "Too Many Empty Chunks", time.time()
                        )
                        continue
                    time.sleep(0.01)  # Small pause for empty chunks
                    continue

                self.consecutive_empty_chunks = 0
                byte_buffer.extend(chunk)

                # Buffer size management
                if len(byte_buffer) > MAX_MJPEG_BUFFER_SIZE:
                    logger.warning(
                        f"MJPEG buffer for {self.mjpeg_url} exceeding {MAX_MJPEG_BUFFER_SIZE} bytes, trimming."
                    )
                    # Trim from the beginning, keeping the end part
                    del byte_buffer[:-MJPEG_BUFFER_KEEP_SIZE_ON_TRIM]

                # Process all complete JPEGs in the buffer
                while True:
                    if self.shutdown_event.is_set():
                        break

                    jpg_start = byte_buffer.find(b"\xff\xd8")  # SOI
                    if jpg_start == -1:
                        break  # No start of a new frame

                    jpg_end = byte_buffer.find(b"\xff\xd9", jpg_start + 2)  # EOI
                    if jpg_end == -1:
                        # Incomplete frame in buffer. If jpg_start > 0, there's leading garbage.
                        if jpg_start > 0:
                            # logger.debug(f"Trimming {jpg_start} bytes of leading garbage from buffer for {self.mjpeg_url}.")
                            del byte_buffer[:jpg_start]
                        break

                    frame_bytes = byte_buffer[jpg_start : jpg_end + 2]
                    del byte_buffer[: jpg_end + 2]  # Consume frame from buffer

                    try:
                        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                        if frame is None:
                            self.consecutive_decode_errors += 1
                            logger.warning(
                                f"Stream {self.mjpeg_url}: Decode error (frame is None). Consecutive: {self.consecutive_decode_errors}"
                            )
                            if (
                                self.consecutive_decode_errors
                                >= MAX_CONSECUTIVE_DECODE_ERRORS
                            ):
                                logger.error(
                                    f"Stream {self.mjpeg_url}: Max decode errors reached. Reconnecting."
                                )
                                self.active = False  # Trigger reconnect
                                self.frame_cache.update_status(
                                    "Max Decode Errors", time.time()
                                )
                                break  # Break from inner jpeg processing loop
                            continue  # Try next jpeg in buffer

                        self.consecutive_decode_errors = 0
                        self.total_frames_decoded += 1
                        current_time = time.time()
                        self.frame_cache.update_frame(frame, current_time, "Running")
                        # logger.debug(f"Frame decoded from {self.mjpeg_url} at {current_time}")
                        retry_delay = 1.0  # Reset retry delay on successful frame

                    except Exception as e_decode:
                        self.consecutive_decode_errors += 1
                        logger.error(
                            f"Stream {self.mjpeg_url}: Exception decoding frame: {e_decode}. Consecutive: {self.consecutive_decode_errors}"
                        )
                        if (
                            self.consecutive_decode_errors
                            >= MAX_CONSECUTIVE_DECODE_ERRORS
                        ):
                            logger.error(
                                f"Stream {self.mjpeg_url}: Max decode errors (exception) reached. Reconnecting."
                            )
                            self.active = False  # Trigger reconnect
                            self.frame_cache.update_status(
                                "Max Decode Errors (Exception)", time.time()
                            )
                            break  # Break from inner jpeg processing loop

                if not self.active:
                    continue  # If reconnect was triggered from inner loop

            except (
                StopIteration
            ):  # next(self._stream) can raise this if stream closes unexpectedly
                logger.warning(
                    f"Stream {self.mjpeg_url} connection closed by server (StopIteration). Attempting to reconnect."
                )
                self.active = False
                self.frame_cache.update_status("Server Closed Connection", time.time())
            except requests.exceptions.ChunkedEncodingError as e:
                logger.warning(
                    f"Stream {self.mjpeg_url} chunked encoding error: {e}. Attempting to reconnect."
                )
                self.active = False
                self.frame_cache.update_status("Chunked Encoding Error", time.time())
            except requests.exceptions.RequestException as e:
                logger.error(
                    f"Stream {self.mjpeg_url} request exception in read loop: {e}. Attempting to reconnect."
                )
                self.active = False
                self.frame_cache.update_status(
                    f"Request Exception: {type(e).__name__}", time.time()
                )
            except Exception as e:
                logger.exception(
                    f"Stream {self.mjpeg_url}: Unexpected error in read frames loop: {e}. Attempting to reconnect."
                )
                self.active = False  # Trigger reconnect
                self.frame_cache.update_status(
                    f"Unexpected Read Loop Error: {type(e).__name__}", time.time()
                )

            if not self.active and not self.shutdown_event.is_set():
                # If loop decided to reconnect (active=False) but not shutting down
                logger.info(
                    f"Reader loop for {self.mjpeg_url} preparing to retry after error. Delay: {retry_delay:.1f}s"
                )
                self.shutdown_event.wait(
                    retry_delay
                )  # Wait for retry_delay or shutdown
                retry_delay = min(retry_delay * 1.5, max_retry_delay)

        logger.info(f"Reader thread for {self.mjpeg_url} terminated.")
        with self.lock:  # Ensure request is closed if thread exits
            if self._request:
                try:
                    self._request.close()
                except:
                    pass
            self._request = None
            self._stream = None
            self.active = False  # Ensure marked inactive
            if (
                not self.frame_cache.get_data().get("status") == "Stopped"
            ):  # Avoid overwriting explicit stop status
                self.frame_cache.update_status("Reader Thread Terminated", time.time())

    def _monitor_idle_loop(self) -> None:
        """Monitors stream activity and stops the stream if idle for too long."""
        logger.info(
            f"Idle monitor thread started for {self.mjpeg_url} (timeout: {self.idle_timeout}s)."
        )
        while not self.shutdown_event.is_set():
            try:
                # Check every second or so.
                # shutdown_event.wait is interruptible by shutdown_event.set()
                if self.shutdown_event.wait(timeout=max(1.0, self.idle_timeout / 10)):
                    break  # Shutdown was signaled

                with self.lock:  # Access self.active and self.last_activity_time
                    if self.active:  # Only monitor if considered active
                        idle_duration = time.time() - self.last_activity_time
                        if idle_duration > self.idle_timeout:
                            logger.info(
                                f"Stream {self.mjpeg_url} idle for {idle_duration:.2f}s (limit: {self.idle_timeout}s). Stopping stream."
                            )
                            # Call self.stop() which acquires lock and handles thread shutdown
                            # To avoid deadlock, call stop outside this lock or ensure stop() is fully re-entrant
                            # For now, assume stop() handles RLock correctly.
                            # A better way might be to set a flag that _read_frames_loop checks,
                            # or just set self.active = False and let _read_frames_loop handle the actual stopping.
                            # Let's make it simpler: signal the reader thread to stop due to idle.
                            self.frame_cache.update_status("Idle Timeout", time.time())
                            self.stop()  # stop() will set shutdown_event and handle cleanup.

            except Exception as e:
                logger.exception(f"Error in idle monitor for {self.mjpeg_url}: {e}")
                # Avoid busy-looping on repeated errors in monitor
                if self.shutdown_event.wait(timeout=5.0):
                    break

        logger.info(f"Idle monitor thread for {self.mjpeg_url} terminated.")

    def health_check(self) -> Dict[str, Any]:
        """Provides health status of the MJPEG stream."""
        cache_data = self.frame_cache.get_data()
        frame_age = (
            (time.time() - cache_data["timestamp"])
            if cache_data["timestamp"] > 0
            else INFINITE_TIMEOUT
        )
        resolution_str = "N/A"
        if cache_data["shape"] and len(cache_data["shape"]) >= 2:
            resolution_str = f"{cache_data['shape'][1]}x{cache_data['shape'][0]}"  # WxH

        return {
            "mjpeg_url": self.mjpeg_url,
            "is_active": self.active,  # Whether the reader *thinks* it should be active
            "stream_status_message": cache_data["status"],
            "frame_timestamp": cache_data["timestamp"],
            "frame_timestamp_str": cache_data["timestamp_str"],
            "frame_age_seconds": frame_age,
            "frame_resolution": resolution_str,
            "last_stream_activity_epoch": getattr(self, "last_activity_time", 0.0),
            "consecutive_decode_errors": self.consecutive_decode_errors,
            "consecutive_empty_chunks": self.consecutive_empty_chunks,
            "total_frames_decoded": self.total_frames_decoded,
        }

    def update_mjpeg_url(self, new_url: str) -> None:
        """
        Updates the MJPEG stream URL. This method is blocking.
        It stops the current stream, updates the URL, and attempts to start the new stream.
        If starting the new stream fails, it attempts to revert to the previously active URL.
        Raises MJPEGStreamError if the new URL is invalid, or if the new URL fails and reverting also fails.
        Raises MJPEGStreamError with a specific message if new URL fails but revert succeeds.
        """
        try:
            validated_new_url = ParameterValidator.validate_mjpeg_url(new_url)
        except ValidationError as e:
            logger.error(
                f"Cannot update MJPEG URL: New URL '{new_url}' is invalid: {e}"
            )
            raise MJPEGStreamError(f"New MJPEG URL '{new_url}' is invalid: {e}") from e

        with self.lock:  # Ensure atomicity of this operation sequence
            current_active_url = (
                self._active_url_before_update
                if self._active_url_before_update
                else self.mjpeg_url
            )

            logger.info(
                f"Attempting to update MJPEG URL from {self.mjpeg_url} to {validated_new_url}."
            )

            if self.active:  # Stop current stream if it's running
                self.stop()  # This will set self.active to False

            self.mjpeg_url = validated_new_url  # Set to new URL

            if self.start():  # Try to start with the new URL
                logger.info(
                    f"Successfully started stream with new URL: {validated_new_url}."
                )
                # self._active_url_before_update is now validated_new_url (set by successful start())
                self.frame_cache.update_status(f"Running on new URL", time.time())
                return  # Success
            else:
                # New URL failed. Attempt to revert.
                logger.error(
                    f"Failed to start stream with new URL: {validated_new_url}. Attempting to revert to: {current_active_url}."
                )
                self.mjpeg_url = current_active_url  # Revert to the stored "good" URL

                if self.start():  # Try to start with the reverted URL
                    logger.warning(
                        f"Successfully reverted and started stream with previous URL: {current_active_url}."
                    )
                    self.frame_cache.update_status(
                        f"Running on reverted URL", time.time()
                    )
                    raise MJPEGStreamError(
                        f"Failed to use new URL '{validated_new_url}'. Stream reverted to '{current_active_url}' and is active."
                    )
                else:
                    # Revert also failed. Stream is now stopped.
                    logger.critical(
                        f"CRITICAL: Failed to start with new URL '{validated_new_url}' AND failed to revert to previous URL '{current_active_url}'. Stream is stopped."
                    )
                    self._active_url_before_update = (
                        None  # No known good URL after this failure
                    )
                    self.frame_cache.update_status(
                        f"Update & Revert Failed", time.time()
                    )
                    raise MJPEGStreamError(
                        f"Failed to use new URL '{validated_new_url}' and also failed to revert to '{current_active_url}'. Stream is inactive."
                    )

    def __del__(self):
        """Destructor to ensure resources are released."""
        logger.debug(
            f"MJPEGStreamReader for {self.mjpeg_url} is being deleted. Ensuring stop."
        )
        if not self.shutdown_event.is_set():  # If not already explicitly stopped
            self.stop()
