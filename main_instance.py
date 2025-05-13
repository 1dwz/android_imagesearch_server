#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Instance Server Script
Starts a single instance of the image matching API server with the specified configuration
"""
import argparse
import logging
import sys

import uvicorn

from modules.api import create_app
from modules.app_config import AppConfig
from modules.constants import SERVER_VERSION
from modules.utils import parse_ini_file

# Configure logger
logger_root = logging.getLogger()
logger_root.setLevel(logging.INFO)

# 创建格式化器
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# 确保控制台输出使用 UTF-8 编码 (特别是 Windows)
try:
    if sys.stdout.encoding != "utf-8":
        # 重新打开 sys.stdout 并指定编码
        sys.stdout = open(
            sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1
        )  # buffering=1 for line buffering
        # 如果需要，也可以对 stderr 进行同样的操作
        # sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
except Exception as e:
    # 如果发生错误（例如，在不支持 fileno() 的环境中），记录警告但继续
    # 在某些交互式环境中可能会出现问题
    logger_root.warning(f"无法设置控制台输出编码为 UTF-8: {e}. 日志输出可能出现乱码。")

# 控制台处理器，现在应该使用 UTF-8 输出
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger_root.addHandler(console_handler)

logger = logging.getLogger(__name__)


def check_port_available(host, port):
    """Check if the given port is available."""
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result != 0


def main():
    """Main entry point for the server startup process."""
    parser = argparse.ArgumentParser(
        description="Start a single instance of the Image Matching API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--port", type=int, required=True, help="Port to run the server on"
    )
    parser.add_argument("--mjpeg-url", required=True, help="URL of the MJPEG stream")
    parser.add_argument(
        "--server-config",
        required=True,
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

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    logger.info(f"Starting image matching server instance (Version {SERVER_VERSION})")
    logger.info(f"Port: {args.port}, Host: {args.host}, MJPEG URL: {args.mjpeg_url}")

    # Check if port is available
    if not check_port_available(args.host, args.port):
        logger.error(f"Port {args.port} is already in use. Exiting.")
        return 1

    try:
        # Parse server configuration file
        logger.info(f"Loading server configuration from {args.server_config}")
        server_config = parse_ini_file(args.server_config)
        if not server_config:
            logger.error(
                f"Failed to parse server configuration file: {args.server_config}"
            )
            return 1

        # Create AppConfig instance
        config = AppConfig(
            mjpeg_url=args.mjpeg_url,
            server_config=server_config,
            port=args.port,
            host=args.host,
        )

        # Create FastAPI application
        app = create_app(config)

        # Start the server
        logger.info(f"Starting server at http://{args.host}:{args.port}")
        uvicorn.run(
            app, host=args.host, port=args.port, log_level=args.log_level.lower()
        )

        return 0
    except Exception as e:
        logger.exception(f"Error starting server: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
