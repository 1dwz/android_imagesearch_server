#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Matching API Server - Simplified Main Script
Provides HTTP API endpoints for template-based image matching in MJPEG streams.
"""
import argparse
import logging
import os
import sys
import uvicorn

from modules.constants import (
    SERVER_VERSION,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MJPEG_URL,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_WAKEUP_TIMEOUT,
    DEFAULT_FILTER_TYPE,
    DEFAULT_MATCH_METHOD,
    DEFAULT_THRESHOLD,
    DEFAULT_CANNY_T1,
    DEFAULT_CANNY_T2,
    DEFAULT_TEMPLATE_CACHE_SIZE,
    DEFAULT_DEBUG_SAVE_DIR,
    DEFAULT_MAX_DEBUG_FILES
)
from modules.api import create_app

# Configure logger
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for the server configuration."""
    parser = argparse.ArgumentParser(
        description="Start the Image Matching API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--mjpeg-url", 
        default=DEFAULT_MJPEG_URL, 
        help="URL of the MJPEG stream"
    )
    
    # Server configuration
    parser.add_argument(
        "--host", 
        default=DEFAULT_HOST, 
        help="Host to bind the server"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=DEFAULT_PORT, 
        help="Port to bind the server"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file", 
        default="server.log", 
        help="Log file path"
    )
    
    # Image processing settings
    parser.add_argument(
        "--default-filter-type",
        default=DEFAULT_FILTER_TYPE,
        choices=["none", "canny"],
        help="Default image filter type"
    )
    parser.add_argument(
        "--default-match-method",
        default=DEFAULT_MATCH_METHOD,
        choices=["ccoeff_normed", "sqdiff_normed", "ccorr_normed"],
        help="Default matching method"
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Default matching threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--default-canny-t1", 
        type=int, 
        default=DEFAULT_CANNY_T1, 
        help="Default Canny threshold 1"
    )
    parser.add_argument(
        "--default-canny-t2",
        type=int,
        default=DEFAULT_CANNY_T2,
        help="Default Canny threshold 2"
    )
    
    # MJPEG stream settings
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=DEFAULT_IDLE_TIMEOUT,
        help="Timeout in seconds for the stream to be considered idle"
    )
    parser.add_argument(
        "--wakeup-timeout",
        type=float,
        default=DEFAULT_WAKEUP_TIMEOUT,
        help="Timeout in seconds to wait for the stream to become ready"
    )
    
    # Cache settings
    parser.add_argument(
        "--template-cache-size",
        type=int,
        default=DEFAULT_TEMPLATE_CACHE_SIZE,
        help="Maximum number of templates to keep in cache"
    )
    
    # Debug settings
    parser.add_argument(
        "--debug-save-dir",
        default=DEFAULT_DEBUG_SAVE_DIR,
        help="Directory to save debug images"
    )
    parser.add_argument(
        "--enable-debug-saving",
        action="store_true",
        help="Enable saving debug images"
    )
    parser.add_argument(
        "--max-debug-files",
        type=int,
        default=DEFAULT_MAX_DEBUG_FILES,
        help="Maximum number of debug image files to retain"
    )
    
    # Other settings
    parser.add_argument(
        "--no-auto-shutdown",
        action="store_true",
        help="Disable automatic shutdown of existing instances"
    )
    parser.add_argument(
        "--skip-dir",
        action="append",
        default=[],
        help="Additional directories to skip in cleanup"
    )
    
    return parser.parse_args()

def configure_logging(level, log_file=None):
    """Configure logging for the application."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Configure handlers
    handlers = []
    
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {e}")
            
    # Always add console handler as fallback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # Configure basic logging
    logging.basicConfig(
        level=numeric_level, 
        handlers=handlers
    )
    
    # Configure uvicorn loggers
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        u_logger = logging.getLogger(name)
        u_logger.propagate = False
        u_logger.handlers = handlers.copy()

def check_port_available(host, port):
    """Check if the given port is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result != 0

def main():
    """Main entry point for the server startup process."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure logging
        configure_logging(args.log_level, args.log_file)
        logger.info(f"Starting Image Matching API Server (Version {SERVER_VERSION})")
        
        # Calculate port based on MJPEG URL (as described in README)
        try:
            # Only calculate dynamic port if port is not explicitly set from command line
            if args.port == DEFAULT_PORT:  # Only apply if the port is still the default value
                from urllib.parse import urlparse
                parsed_url = urlparse(args.mjpeg_url)
                ip_parts = parsed_url.netloc.split(':')[0].split('.')
                if len(ip_parts) == 4:
                    dynamic_port = 60000 + int(ip_parts[3])
                    logger.info(f"Dynamically calculated port based on MJPEG URL: {dynamic_port}")
                    args.port = dynamic_port
        except Exception as e:
            logger.warning(f"Could not calculate port from MJPEG URL: {e}")
            
        # Check if port is available
        if not check_port_available(args.host, args.port):
            logger.error(f"Port {args.port} is already in use. Exiting.")
            return 1
            
        # Create FastAPI application
        app = create_app(args)
        
        # Start the server
        logger.info(f"Starting server at http://{args.host}:{args.port}")
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            log_level=args.log_level.lower()
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested (KeyboardInterrupt)")
        return 0
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
