# -*- coding: utf-8 -*-
"""
MJPEG Stream Reader Module
Handles connection to MJPEG streams, frame capturing, and stream management
"""
import logging
import threading
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests

from modules.constants import DEFAULT_IDLE_TIMEOUT, DEFAULT_WAKEUP_TIMEOUT

logger = logging.getLogger(__name__)


class MJPEGStreamReader:
    """
    Handles reading frames from an MJPEG stream with support for
    idle timeouts and automatic reconnection.
    """

    def __init__(
        self,
        mjpeg_url: str,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        wakeup_timeout: float = DEFAULT_WAKEUP_TIMEOUT,
    ):
        """
        Initialize the MJPEG stream reader.

        Args:
            mjpeg_url: URL of the MJPEG stream
            idle_timeout: Seconds of inactivity before the stream connection is closed
            wakeup_timeout: Maximum seconds to wait for stream to be ready on wake-up
        """
        self.mjpeg_url = mjpeg_url
        self.idle_timeout = idle_timeout
        self.wakeup_timeout = wakeup_timeout

        # Stream state
        self.stream = None
        self.active = False
        self.last_activity = 0

        # Current frame state
        self.current_frame = None
        self.frame_timestamp = 0
        self.frame_dimensions = (0, 0)  # width, height

        # Thread synchronization
        self.lock = threading.RLock()
        self.idle_monitor_thread = None
        self.reader_thread = None
        self.shutdown_event = threading.Event()

        logger.info(f"MJPEG Stream Reader initialized with URL: {mjpeg_url}")
        logger.info(f"Idle timeout: {idle_timeout}s, Wakeup timeout: {wakeup_timeout}s")

    def start(self) -> bool:
        """
        Start the MJPEG stream reader.

        Returns:
            bool: True if successfully started, False otherwise
        """
        with self.lock:
            if self.active:
                logger.debug("Stream already active, ignoring start request")
                return True

            try:
                logger.info("Connecting to MJPEG stream...")
                self.stream = requests.get(self.mjpeg_url, stream=True, timeout=10)
                self.stream.raise_for_status()

                content_type = self.stream.headers.get("Content-Type", "")
                if "multipart/x-mixed-replace" not in content_type:
                    logger.error(f"Unexpected content type: {content_type}")
                    self.stream.close()
                    return False

                logger.info("Connected to MJPEG stream")
                self.active = True
                self.last_activity = time.time()

                # Start reader thread
                self.shutdown_event.clear()
                self.reader_thread = threading.Thread(
                    target=self._read_frames, daemon=True, name="MJPEGReaderThread"
                )
                self.reader_thread.start()

                # Start idle monitor thread
                self.idle_monitor_thread = threading.Thread(
                    target=self._monitor_idle,
                    daemon=True,
                    name="MJPEGIdleMonitorThread",
                )
                self.idle_monitor_thread.start()

                return True

            except Exception as e:
                logger.error(f"Failed to connect to MJPEG stream: {e}")
                self.active = False
                if self.stream:
                    self.stream.close()
                    self.stream = None
                return False

    def stop(self) -> None:
        """Stop the MJPEG stream reader."""
        with self.lock:
            if not self.active:
                return

            logger.info("Stopping MJPEG stream reader...")
            self.shutdown_event.set()
            self.active = False

            if self.stream:
                try:
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Error closing stream: {e}")
                finally:
                    self.stream = None

            # Wait for threads to finish
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(2.0)  # Wait up to 2 seconds

            if self.idle_monitor_thread and self.idle_monitor_thread.is_alive():
                self.idle_monitor_thread.join(2.0)  # Wait up to 2 seconds

            logger.info("MJPEG stream reader stopped")

    def get_frame(self) -> Tuple[Optional[np.ndarray], float, Tuple[int, int]]:
        """
        Get the most recent frame from the stream.

        Returns:
            Tuple containing:
            - The frame as a numpy array, or None if no frame is available
            - The timestamp of the frame
            - The dimensions (width, height) of the frame
        """
        with self.lock:
            # Update last activity time
            self.last_activity = time.time()

            # Start the stream if it's not active
            if not self.active:
                logger.info("Stream not active, attempting to start")
                success = self.start()

                # Wait for the stream to become ready
                start_time = time.time()
                while success and self.current_frame is None:
                    if time.time() - start_time > self.wakeup_timeout:
                        logger.error("Timed out waiting for stream to provide frames")
                        return None, 0, (0, 0)
                    time.sleep(0.1)  # Short sleep to avoid busy waiting

            return self.current_frame, self.frame_timestamp, self.frame_dimensions

    def _read_frames(self) -> None:
        """
        Read frames from the MJPEG stream in a background thread.
        """
        if not self.stream:
            logger.error("Cannot read frames: stream is not initialized")
            return

        bytes_buffer = bytes()

        try:
            for chunk in self.stream.iter_content(chunk_size=1024):
                if self.shutdown_event.is_set():
                    logger.debug("Shutdown event detected in reader thread")
                    break

                if not chunk:
                    continue

                bytes_buffer += chunk
                jpg_start = bytes_buffer.find(b"\xff\xd8")
                jpg_end = bytes_buffer.find(b"\xff\xd9")

                if jpg_start != -1 and jpg_end != -1:
                    # Extract the JPEG frame
                    jpg_data = bytes_buffer[jpg_start : jpg_end + 2]
                    bytes_buffer = bytes_buffer[jpg_end + 2 :]

                    # Decode the frame
                    try:
                        frame = cv2.imdecode(
                            np.frombuffer(jpg_data, np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            with self.lock:
                                self.current_frame = frame
                                self.frame_timestamp = time.time()
                                self.frame_dimensions = (
                                    frame.shape[1],
                                    frame.shape[0],
                                )  # width, height
                    except Exception as e:
                        logger.error(f"Error decoding frame: {e}")

        except Exception as e:
            if not self.shutdown_event.is_set():  # Only log if not shutting down
                logger.error(f"Error in MJPEG reader thread: {e}")

        logger.debug("MJPEG reader thread exiting")

    def _monitor_idle(self) -> None:
        """
        Monitor for idle connections and close the stream if idle for too long.
        """
        while not self.shutdown_event.is_set():
            time.sleep(1.0)  # Check every second

            with self.lock:
                if not self.active:
                    continue

                idle_time = time.time() - self.last_activity
                if idle_time > self.idle_timeout:
                    logger.info(f"Stream idle for {idle_time:.1f}s, closing connection")
                    if self.stream:
                        try:
                            self.stream.close()
                        except Exception as e:
                            logger.warning(f"Error closing idle stream: {e}")
                        finally:
                            self.stream = None
                    self.active = False

        logger.debug("MJPEG idle monitor thread exiting")

    def health_check(self) -> Dict:
        """
        Check the health of the MJPEG stream.

        Returns:
            Dict with health status information
        """
        with self.lock:
            now = time.time()
            frame_age = (
                now - self.frame_timestamp if self.frame_timestamp > 0 else 9999999.0
            )

            return {
                "active": self.active,
                "frame_valid": self.current_frame is not None,
                "frame_age_seconds": frame_age,
                "frame_resolution": f"{self.frame_dimensions[0]}x{self.frame_dimensions[1]}",
                "last_activity_seconds_ago": (
                    now - self.last_activity if self.last_activity > 0 else 9999999.0
                ),
            }

    def __del__(self):
        """Ensure resources are properly released on deletion."""
        self.stop()
