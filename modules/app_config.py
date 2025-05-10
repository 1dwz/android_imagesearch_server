# -*- coding: utf-8 -*-
"""
AppConfig Module
Defines the configuration class used to create and configure FastAPI application instances
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class AppConfig:
    """
    Holds configuration and state for one stream instance.
    This class is used to decouple the global state from the application instance,
    allowing multiple instances to be created with different configurations.
    """

    def __init__(
        self,
        mjpeg_url: str,
        server_config: Dict[str, Any],
        port: int,
        host: str = "127.0.0.1",
    ):
        """
        Initialize a new AppConfig instance.

        Args:
            mjpeg_url: URL of the MJPEG stream to connect to
            server_config: Dictionary containing server configuration options
            port: Port to run the server on
            host: Host to bind the server to
        """
        self.mjpeg_url = mjpeg_url
        self.server_config = server_config
        self.port = port
        self.host = host

        # Initialize stream reader and image processor references to None
        # These will be set when the app is created
        self.stream_reader = None
        self.image_processor = None

        logger.debug(f"Created AppConfig for MJPEG URL: {mjpeg_url}, port: {port}")

    def get_stream_config(self) -> Dict[str, Any]:
        """
        Get stream configuration parameters from the server config.

        Returns:
            Dictionary containing stream configuration parameters
        """
        return {
            "mjpeg_url": self.mjpeg_url,
            "idle_timeout": self.server_config.get("idle_timeout", 300.0),
            "wakeup_timeout": self.server_config.get("wakeup_timeout", 15.0),
        }

    def get_image_processor_config(self) -> Dict[str, Any]:
        """
        Get image processor configuration parameters from the server config.

        Returns:
            Dictionary containing image processor configuration parameters
        """
        # Just return the entire server config for now
        # In a more refined implementation, you might want to filter this
        return self.server_config
