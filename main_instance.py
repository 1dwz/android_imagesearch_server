#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Instance Server Script
Starts a single instance of the image matching API server with the specified configuration
"""
import argparse
import configparser  # Import configparser
import logging
import sys
from typing import Any, Dict  # For type hinting

import uvicorn

from modules.api import create_app
from modules.app_config import AppConfig
from modules.constants import (  # Import specific defaults for parsing; Add DEFAULT_INI_CACHE_SIZE if you plan to use it from constants.py
    DEFAULT_CANNY_T1,
    DEFAULT_CANNY_T2,
    DEFAULT_FILTER_TYPE,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_MATCH_METHOD,
    DEFAULT_MJPEG_URL,
    DEFAULT_PORT,
    DEFAULT_SERVER_CONFIG_PATH,
    DEFAULT_TEMPLATE_CACHE_SIZE,
    DEFAULT_THRESHOLD,
    DEFAULT_WAKEUP_TIMEOUT,
    SERVER_VERSION,
)

# Import parse_ini_file and is_port_in_use from utils
from modules.utils import is_port_in_use

logger = logging.getLogger(__name__)


def parse_server_config_ini(config_path: str) -> Dict[str, Any]:
    """Parses the server.ini configuration file into a dictionary with typed values."""
    cp = configparser.ConfigParser()
    read_files = cp.read(config_path, encoding="utf-8")
    if not read_files:
        logger.error(f"Server configuration file not found or empty: {config_path}")
        raise FileNotFoundError(f"Server configuration file not found: {config_path}")

    parsed_config: Dict[str, Any] = {}

    # Helper to get typed values
    def get_typed_option(section: str, option: str, type_func, default_val=None):
        if cp.has_option(section, option):
            try:
                return type_func(cp.get(section, option))
            except ValueError:
                logger.warning(
                    f"Invalid value for {option} in section {section} of {config_path}. Using default: {default_val}"
                )
                return default_val
        return default_val

    # Server section (host is usually from CLI args, but can be a default here)
    # parsed_config['host'] = cp.get('server', 'host', fallback='127.0.0.1') # Example

    # Debug section
    if cp.has_section("debug"):
        parsed_config["enable_debug_saving"] = cp.getboolean(
            "debug", "enable_debug_saving", fallback=False
        )
        parsed_config["debug_save_dir"] = cp.get(
            "debug", "debug_save_dir", fallback="debug_images"
        )
        parsed_config["max_debug_files"] = cp.getint(
            "debug", "max_debug_files", fallback=100
        )

    # Timeout section
    if cp.has_section("timeout"):
        parsed_config["idle_timeout"] = cp.getfloat(
            "timeout", "idle_timeout", fallback=DEFAULT_IDLE_TIMEOUT
        )
        parsed_config["wakeup_timeout"] = cp.getfloat(
            "timeout", "wakeup_timeout", fallback=DEFAULT_WAKEUP_TIMEOUT
        )

    # MatchSettings section (for global defaults for ImageProcessor)
    if cp.has_section("MatchSettings"):
        parsed_config["default_filter_type"] = cp.get(
            "MatchSettings", "filter_type", fallback=DEFAULT_FILTER_TYPE
        )
        parsed_config["default_match_method"] = cp.get(
            "MatchSettings", "match_method", fallback=DEFAULT_MATCH_METHOD
        )
        parsed_config["default_threshold"] = cp.getfloat(
            "MatchSettings", "threshold", fallback=DEFAULT_THRESHOLD
        )
        parsed_config["default_canny_t1"] = cp.getint(
            "MatchSettings", "canny_t1", fallback=DEFAULT_CANNY_T1
        )
        parsed_config["default_canny_t2"] = cp.getint(
            "MatchSettings", "canny_t2", fallback=DEFAULT_CANNY_T2
        )

    # Cache section
    if cp.has_section("cache"):
        parsed_config["template_cache_size"] = cp.getint(
            "cache", "template_cache_size", fallback=DEFAULT_TEMPLATE_CACHE_SIZE
        )
        # ini_cache_size from server.ini might be intended for the global utils.ini_cache
        # However, utils.ini_cache is initialized globally. Modifying it per instance is complex.
        # If this setting is truly per-instance, INICache should not be global.
        # For now, we load it, but AppConfig/ImageProcessor might not use it if INICache is not instance-specific.
        if cp.has_option("cache", "ini_cache_size"):
            parsed_config["ini_cache_size"] = cp.getint(
                "cache", "ini_cache_size"
            )  # No easy constant default for this here

    logger.info(f"Successfully parsed server config from {config_path}")
    return parsed_config


def main():
    """Main entry point for the server instance."""
    parser = argparse.ArgumentParser(
        description="Start a single instance of the Image Matching API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=DEFAULT_PORT,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--mjpeg-url",
        required=False,
        default=DEFAULT_MJPEG_URL,
        help="URL of the MJPEG stream",
    )
    parser.add_argument(
        "--server-config",
        required=False,  # Make it optional if we provide defaults
        default=DEFAULT_SERVER_CONFIG_PATH,
        help="Path to the server configuration file (INI format)",
    )

    # Optional arguments
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set log level for this application's logger (Uvicorn will have its own)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"Starting image matching server instance (Version {SERVER_VERSION})")
    logger.info(
        f"Config: Port={args.port}, Host={args.host}, MJPEG_URL={args.mjpeg_url}, ServerINI={args.server_config}"
    )

    if is_port_in_use(args.port, args.host):  # Check against specified host
        logger.error(
            f"Port {args.port} on host {args.host} is already in use. Exiting."
        )
        return 1

    try:
        # Parse server configuration file
        logger.info(f"Loading server configuration from: {args.server_config}")
        server_settings_from_ini = parse_server_config_ini(args.server_config)

        # Create AppConfig instance
        # AppConfig will use values from server_settings_from_ini,
        # and its internal .get() calls will use its own defaults if a key is missing.
        app_config_instance = AppConfig(
            mjpeg_url=args.mjpeg_url,
            server_config=server_settings_from_ini,
            port=args.port,
            host=args.host,
        )

        # Create FastAPI application
        app = create_app(app_config_instance)

        # Start the server
        logger.info(f"Starting Uvicorn server at http://{args.host}:{args.port}")
        uvicorn.run(
            app, host=args.host, port=args.port, log_level=args.log_level.lower()
        )

        return 0
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Error starting server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
