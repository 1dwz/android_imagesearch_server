#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Instance Server Manager
Manages multiple instances of the image matching API server
"""
import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, List

import requests
from requests.exceptions import ConnectionError, Timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_MULTI_CONFIG_PATH = "multi_config.json"
DEFAULT_SERVER_CONFIG_PATH = "config/server.ini"

# Command timeout settings
REQUEST_TIMEOUT = 2.0  # seconds
PROCESS_START_TIMEOUT = 15.0  # seconds


def is_port_in_use(port: int) -> bool:
    """
    Check if the specified port is in use.

    Args:
        port: Port number to check

    Returns:
        True if the port is in use, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.warning(f"Error checking port {port}: {e}")
        return False


def get_instance_status(port: int) -> Dict[str, Any]:
    """
    Get the status of a server instance running on the specified port.

    Args:
        port: Port number of the instance

    Returns:
        Dictionary containing status information
    """
    url = f"http://127.0.0.1:{port}/internal/status"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            status_data = response.json()
            logger.debug(f"Status for port {port}: {status_data}")
            return {"port": port, "status": "running", "details": status_data}
        else:
            logger.warning(
                f"Instance on port {port} returned status code {response.status_code}"
            )
            return {
                "port": port,
                "status": "error",
                "message": f"API returned status code {response.status_code}",
            }
    except ConnectionError:
        if is_port_in_use(port):
            logger.warning(f"Port {port} is in use but API is not responding")
            return {
                "port": port,
                "status": "unresponsive",
                "message": "Port is in use but API is not responding",
            }
        else:
            logger.debug(f"No server running on port {port}")
            return {
                "port": port,
                "status": "stopped",
                "message": "No server running on this port",
            }
    except Timeout:
        logger.warning(f"Timeout while connecting to instance on port {port}")
        return {"port": port, "status": "timeout", "message": "Connection timed out"}
    except Exception as e:
        logger.error(f"Error getting status from port {port}: {e}")
        return {"port": port, "status": "error", "message": str(e)}


def start_instance(
    port: int,
    mjpeg_url: str,
    server_config_path: str,
    host: str = "127.0.0.1",
    log_level: str = "info",
) -> Dict[str, Any]:
    """
    Start a server instance on the specified port.

    Args:
        port: Port to run the server on
        mjpeg_url: URL of the MJPEG stream
        server_config_path: Path to the server configuration file
        host: Host to bind the server to
        log_level: Logging level for the server

    Returns:
        Dictionary containing the start result
    """
    # First check if an instance is already running on this port
    if is_port_in_use(port):
        status = get_instance_status(port)
        if status["status"] in ["running", "unresponsive"]:
            logger.info(f"Server already running on port {port}")
            return {"port": port, "status": "already_running", "details": status}
        else:
            logger.warning(f"Port {port} is in use but not by our server")
            return {
                "port": port,
                "status": "port_in_use",
                "message": "Port is in use by another process",
            }

    # Start the server process
    logger.info(f"Starting server instance on port {port} with MJPEG URL: {mjpeg_url}")

    # Build command arguments
    cmd = [
        sys.executable,
        "main_instance.py",
        "--port",
        str(port),
        "--mjpeg-url",
        mjpeg_url,
        "--server-config",
        server_config_path,
        "--host",
        host,
        "--log-level",
        log_level,
    ]

    try:
        # Create log directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        # Prepare log files
        log_file = os.path.join(logs_dir, f"server_{port}.log")

        # Open log file
        with open(log_file, "a") as log_f:
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == "win32"
                    else 0
                ),
            )

            logger.info(f"Started process for port {port} with PID {process.pid}")
            logger.info(f"Logs being written to {log_file}")

            # Wait for the server to start
            start_time = time.time()
            while time.time() - start_time < PROCESS_START_TIMEOUT:
                if is_port_in_use(port):
                    # Give it a moment to initialize the API
                    time.sleep(1.0)
                    # Check if the API is responding
                    status = get_instance_status(port)
                    if status["status"] == "running":
                        logger.info(
                            f"Server instance on port {port} started successfully"
                        )
                        return {
                            "port": port,
                            "status": "started",
                            "pid": process.pid,
                            "details": status["details"] if "details" in status else {},
                        }
                    elif status["status"] == "unresponsive":
                        logger.info(
                            f"Server instance on port {port} started but API not ready yet"
                        )
                        # Keep waiting
                    else:
                        logger.warning(f"Unexpected status for port {port}: {status}")
                        # Keep waiting
                time.sleep(0.5)

            # If we get here, the server didn't start correctly
            logger.error(f"Timeout waiting for server on port {port} to start")

            # Try to get status one more time
            if is_port_in_use(port):
                status = get_instance_status(port)
                return {
                    "port": port,
                    "status": "started_but_unresponsive",
                    "pid": process.pid,
                    "details": status,
                }

            # If we can't connect at all, try to terminate the process
            try:
                process.terminate()
                logger.info(f"Terminated process {process.pid} for port {port}")
            except Exception as term_err:
                logger.error(f"Error terminating process {process.pid}: {term_err}")

            return {
                "port": port,
                "status": "failed_to_start",
                "message": "Server process started but failed to bind to port",
            }
    except Exception as e:
        logger.exception(f"Error starting server instance on port {port}")
        return {"port": port, "status": "error", "message": str(e)}


def update_mjpeg_url(port: int, new_url: str) -> Dict[str, Any]:
    """
    Update the MJPEG URL of a running server instance.

    Args:
        port: Port of the server instance
        new_url: New MJPEG URL

    Returns:
        Dictionary containing the update result
    """
    url = f"http://127.0.0.1:{port}/internal/update_mjpeg_url"

    # First check if the instance is running
    if not is_port_in_use(port):
        logger.error(f"No server running on port {port}")
        return {
            "port": port,
            "status": "error",
            "message": "No server running on this port",
        }

    # Send the update request
    try:
        logger.info(f"Updating MJPEG URL for port {port} to: {new_url}")
        response = requests.put(url, json={"new_url": new_url}, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            logger.info(f"Successfully updated MJPEG URL for port {port}")
            return {
                "port": port,
                "status": "ok",
                "message": f"MJPEG URL updated to {new_url}",
            }
        else:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except Exception:
                error_detail = response.text

            logger.error(f"Failed to update MJPEG URL for port {port}: {error_detail}")
            return {"port": port, "status": "error", "message": error_detail}
    except ConnectionError:
        logger.error(f"Connection refused for port {port}")
        return {"port": port, "status": "error", "message": "Connection refused"}
    except Timeout:
        logger.error(f"Request timed out for port {port}")
        return {"port": port, "status": "error", "message": "Request timed out"}
    except Exception as e:
        logger.exception(f"Error updating MJPEG URL for port {port}")
        return {"port": port, "status": "error", "message": str(e)}


def stop_instance(port: int) -> Dict[str, Any]:
    """
    Stop a running server instance.

    Args:
        port: Port of the server instance

    Returns:
        Dictionary containing the stop result
    """
    # Check if the instance is running
    if not is_port_in_use(port):
        logger.info(f"No server running on port {port}")
        return {
            "port": port,
            "status": "already_stopped",
            "message": "No server running on this port",
        }

    # Get process information
    import psutil

    try:
        # Find process using the port
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.status == "LISTEN":
                pid = conn.pid
                logger.info(f"Found process {pid} using port {port}")

                # Get process object
                process = psutil.Process(pid)

                # Stop the process
                logger.info(f"Stopping process {pid} for port {port}")
                if sys.platform == "win32":
                    # On Windows, send Ctrl+C signal
                    process.terminate()
                else:
                    # On Unix, send SIGTERM
                    process.send_signal(signal.SIGTERM)

                # Wait for the process to terminate
                max_wait = 10.0  # seconds
                wait_interval = 0.5  # seconds
                start_time = time.time()

                while time.time() - start_time < max_wait:
                    if not is_port_in_use(port):
                        logger.info(f"Server on port {port} stopped successfully")
                        return {
                            "port": port,
                            "status": "stopped",
                            "message": "Server stopped successfully",
                        }

                    # Check if process is still running
                    try:
                        process.status()
                        # Process still exists, wait a bit more
                        time.sleep(wait_interval)
                    except psutil.NoSuchProcess:
                        # Process no longer exists
                        if is_port_in_use(port):
                            logger.warning(
                                f"Process ended but port {port} still in use"
                            )
                            break
                        else:
                            logger.info(f"Process {pid} terminated for port {port}")
                            return {
                                "port": port,
                                "status": "stopped",
                                "message": f"Process {pid} terminated",
                            }

                # If we get here, the process didn't terminate gracefully
                logger.warning(
                    f"Timeout waiting for process {pid} to terminate. Killing..."
                )

                try:
                    process.kill()
                    logger.info(f"Killed process {pid} for port {port}")
                    return {
                        "port": port,
                        "status": "killed",
                        "message": f"Process {pid} killed",
                    }
                except Exception as kill_err:
                    logger.error(f"Failed to kill process {pid}: {kill_err}")
                    return {
                        "port": port,
                        "status": "error",
                        "message": f"Failed to kill process: {kill_err}",
                    }

        # If we get here, we didn't find a process for the port
        logger.warning(f"Port {port} is in use but couldn't find the process")
        return {
            "port": port,
            "status": "error",
            "message": "Port is in use but couldn't find the process",
        }
    except Exception as e:
        logger.exception(f"Error stopping server on port {port}")
        return {"port": port, "status": "error", "message": str(e)}


def load_multi_config(config_path: str) -> List[Dict[str, Any]]:
    """
    Load multi-instance configuration from a JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        List of instance configurations
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate config format
        if not isinstance(config, list):
            logger.error(
                f"Invalid configuration format: {config_path} must contain a list"
            )
            return []

        # Validate each item
        valid_configs = []
        for idx, item in enumerate(config):
            if not isinstance(item, dict):
                logger.error(
                    f"Invalid configuration at index {idx}: must be a dictionary"
                )
                continue

            if "port" not in item:
                logger.error(f"Missing 'port' in configuration at index {idx}")
                continue

            if "mjpeg_url" not in item:
                logger.error(f"Missing 'mjpeg_url' in configuration at index {idx}")
                continue

            valid_configs.append(item)

        logger.info(
            f"Loaded {len(valid_configs)} valid instance configurations from {config_path}"
        )
        return valid_configs
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON configuration: {e}")
        return []
    except Exception as e:
        logger.exception(f"Error loading configuration from {config_path}")
        return []


def create_multi_config(config_path: str, instances: List[Dict[str, Any]]) -> bool:
    """
    Create or update multi-instance configuration file.

    Args:
        config_path: Path to the configuration file
        instances: List of instance configurations

    Returns:
        True on success, False on failure
    """
    try:
        with open(config_path, "w") as f:
            json.dump(instances, f, indent=2)

        logger.info(f"Saved {len(instances)} instance configurations to {config_path}")
        return True
    except Exception as e:
        logger.exception(f"Error saving configuration to {config_path}")
        return False


def main():
    """Main entry point for the server manager."""
    parser = argparse.ArgumentParser(
        description="Manage multiple instances of the Image Matching API Server",
    )

    # Global options
    parser.add_argument(
        "--multi-config",
        default=DEFAULT_MULTI_CONFIG_PATH,
        help=f"Path to multi-instance configuration file (default: {DEFAULT_MULTI_CONFIG_PATH})",
    )
    parser.add_argument(
        "--server-config",
        default=DEFAULT_SERVER_CONFIG_PATH,
        help=f"Path to server configuration file (default: {DEFAULT_SERVER_CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start server instances")
    start_parser.add_argument(
        "--port", type=int, help="Start a specific instance by port"
    )
    start_parser.add_argument(
        "--mjpeg-url",
        help="MJPEG URL for the instance (required if --port is specified and port not in config)",
    )

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop server instances")
    stop_parser.add_argument(
        "--port", type=int, help="Stop a specific instance by port"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Get status of server instances"
    )
    status_parser.add_argument(
        "--port", type=int, help="Get status of a specific instance by port"
    )

    # Update command
    update_parser = subparsers.add_parser(
        "update", help="Update server instance configuration"
    )
    update_parser.add_argument(
        "--port", type=int, required=True, help="Port of the instance to update"
    )
    update_parser.add_argument(
        "--new-mjpeg-url", required=True, help="New MJPEG URL for the instance"
    )

    # Create command
    create_parser = subparsers.add_parser(
        "create", help="Create a new server instance configuration"
    )
    create_parser.add_argument(
        "--port", type=int, required=True, help="Port for the new instance"
    )
    create_parser.add_argument(
        "--mjpeg-url", required=True, help="MJPEG URL for the new instance"
    )

    # Delete command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a server instance configuration"
    )
    delete_parser.add_argument(
        "--port", type=int, required=True, help="Port of the instance to delete"
    )

    # Parse arguments
    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Load multi-instance configuration
    multi_config = []
    if os.path.exists(args.multi_config):
        multi_config = load_multi_config(args.multi_config)

    # Execute command
    if args.command == "start":
        # Start instances
        if args.port:
            # Start a specific instance
            instance_config = next(
                (item for item in multi_config if item["port"] == args.port), None
            )

            if instance_config:
                # Instance found in config
                mjpeg_url = instance_config["mjpeg_url"]
                if args.mjpeg_url:
                    # Override MJPEG URL if specified
                    mjpeg_url = args.mjpeg_url
                    logger.info(
                        f"Overriding MJPEG URL for port {args.port}: {mjpeg_url}"
                    )
            elif args.mjpeg_url:
                # Instance not in config but MJPEG URL specified
                mjpeg_url = args.mjpeg_url
                logger.info(
                    f"Starting new instance on port {args.port} with MJPEG URL: {mjpeg_url}"
                )
            else:
                # Instance not in config and no MJPEG URL specified
                logger.error(
                    f"No configuration found for port {args.port} and no MJPEG URL specified"
                )
                return 1

            # Start the instance
            result = start_instance(
                port=args.port,
                mjpeg_url=mjpeg_url,
                server_config_path=args.server_config,
                log_level=args.log_level,
            )

            # Print result
            print(json.dumps(result, indent=2))

            # Update config if instance started successfully and wasn't in config
            if (
                result["status"] in ["started", "already_running"]
                and not instance_config
            ):
                multi_config.append({"port": args.port, "mjpeg_url": mjpeg_url})
                create_multi_config(args.multi_config, multi_config)
        else:
            # Start all instances
            results = []
            for instance in multi_config:
                result = start_instance(
                    port=instance["port"],
                    mjpeg_url=instance["mjpeg_url"],
                    server_config_path=args.server_config,
                    log_level=args.log_level,
                )
                results.append(result)

            # Print results
            print(json.dumps(results, indent=2))

    elif args.command == "stop":
        # Stop instances
        if args.port:
            # Stop a specific instance
            result = stop_instance(args.port)
            print(json.dumps(result, indent=2))
        else:
            # Stop all instances
            results = []
            for instance in multi_config:
                result = stop_instance(instance["port"])
                results.append(result)
            print(json.dumps(results, indent=2))

    elif args.command == "status":
        # Get status
        if args.port:
            # Get status of a specific instance
            result = get_instance_status(args.port)
            print(json.dumps(result, indent=2))
        else:
            # Get status of all instances
            results = []
            for instance in multi_config:
                result = get_instance_status(instance["port"])
                results.append(result)
            print(json.dumps(results, indent=2))

    elif args.command == "update":
        # Update MJPEG URL
        result = update_mjpeg_url(args.port, args.new_mjpeg_url)
        print(json.dumps(result, indent=2))

        # Update config if update was successful
        if result["status"] == "ok":
            instance_config = next(
                (item for item in multi_config if item["port"] == args.port), None
            )
            if instance_config:
                instance_config["mjpeg_url"] = args.new_mjpeg_url
                create_multi_config(args.multi_config, multi_config)

    elif args.command == "create":
        # Create a new instance configuration
        # Check if port already exists in config
        instance_config = next(
            (item for item in multi_config if item["port"] == args.port), None
        )
        if instance_config:
            logger.error(
                f"Instance with port {args.port} already exists in configuration"
            )
            return 1

        # Add new instance to config
        multi_config.append({"port": args.port, "mjpeg_url": args.mjpeg_url})

        # Save config
        if create_multi_config(args.multi_config, multi_config):
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "message": f"Added instance with port {args.port} to configuration",
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    {"status": "error", "message": "Failed to save configuration"},
                    indent=2,
                )
            )

    elif args.command == "delete":
        # Delete an instance configuration
        # Check if port exists in config
        instance_config = next(
            (item for item in multi_config if item["port"] == args.port), None
        )
        if not instance_config:
            logger.error(f"No instance with port {args.port} found in configuration")
            return 1

        # Remove instance from config
        multi_config = [item for item in multi_config if item["port"] != args.port]

        # Save config
        if create_multi_config(args.multi_config, multi_config):
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "message": f"Removed instance with port {args.port} from configuration",
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    {"status": "error", "message": "Failed to save configuration"},
                    indent=2,
                )
            )

    else:
        # No command specified
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
