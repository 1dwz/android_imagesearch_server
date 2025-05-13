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
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional  # Added Optional

import requests
from requests.exceptions import (  # Added RequestException
    ConnectionError,
    RequestException,
    Timeout,
)

from modules.constants import (
    DEFAULT_MULTI_CONFIG_PATH as DEFAULT_MANAGER_MULTI_CONFIG_PATH,  # Import constants used by manager; Status constants; Alias if needed
)
from modules.constants import LOGS_DIR  # For log file paths
from modules.constants import SERVER_LOG_FILENAME_FORMAT  # For log file paths
from modules.constants import (
    DEFAULT_SERVER_CONFIG_PATH,
    PROCESS_START_TIMEOUT,
)
from modules.constants import REQUEST_TIMEOUT as DEFAULT_REQUEST_TIMEOUT
from modules.constants import (
    STATUS_ERROR,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_UNRESPONSIVE,
)

# Import is_port_in_use from modules.utils
from modules.utils import is_port_in_use

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Use constants for timeouts
REQUEST_TIMEOUT_SECONDS = DEFAULT_REQUEST_TIMEOUT
PROCESS_START_TIMEOUT_SECONDS = PROCESS_START_TIMEOUT


def get_instance_status(port: int, host: str = "127.0.0.1") -> Dict[str, Any]:
    """
    Get the status of a server instance running on the specified port and host.
    """
    url = f"http://{host}:{port}/internal/status"
    status_payload: Dict[str, Any] = {
        "port": port,
        "host": host,
    }  # Include host in return

    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 200:
            try:
                status_data = response.json()
                status_payload.update(
                    {
                        "status": STATUS_RUNNING,  # Assuming /internal/status implies running if 200 OK
                        "details": status_data,
                    }
                )
                # Refine status based on details from the instance itself
                if (
                    status_data.get("status") == "error"
                ):  # Check if instance reports itself as error
                    status_payload["status"] = STATUS_ERROR
                    status_payload["message"] = status_data.get(
                        "message", "Instance reported an error."
                    )
                elif status_data.get("status") == "warning":
                    status_payload["status"] = status_data.get(
                        "status"
                    )  # keep 'warning'
                    status_payload["message"] = status_data.get(
                        "message", "Instance reported a warning."
                    )

            except json.JSONDecodeError:
                logger.warning(
                    f"Instance on port {port} (host {host}) returned 200 but with invalid JSON."
                )
                status_payload.update(
                    {
                        "status": STATUS_ERROR,
                        "message": "API returned 200 but with invalid JSON response for status.",
                    }
                )
        else:
            logger.warning(
                f"Instance on port {port} (host {host}) /internal/status returned HTTP {response.status_code}"
            )
            status_payload.update(
                {
                    "status": STATUS_ERROR,
                    "message": f"API /internal/status returned HTTP status code {response.status_code}",
                    "http_status_code": response.status_code,
                }
            )
    except ConnectionError:
        # Port might be in use by something else, or our server not fully up
        if is_port_in_use(port, host):
            logger.warning(
                f"Port {port} (host {host}) is in use, but /internal/status API is not responding (ConnectionError)."
            )
            status_payload.update(
                {
                    "status": STATUS_UNRESPONSIVE,
                    "message": "Port is in use but API /internal/status is not responding (ConnectionError).",
                }
            )
        else:
            logger.debug(
                f"No server found on port {port} (host {host}) (ConnectionError and port not in use)."
            )
            status_payload.update(
                {
                    "status": STATUS_STOPPED,
                    "message": "No server running on this port (ConnectionError and port not in use).",
                }
            )
    except Timeout:
        logger.warning(
            f"Request to /internal/status for port {port} (host {host}) timed out."
        )
        status_payload.update(
            {
                "status": STATUS_UNRESPONSIVE,  # Timeout suggests unresponsive rather than 'timeout' status
                "message": "Connection to API /internal/status timed out.",
            }
        )
    except RequestException as e_req:  # Catch other requests library errors
        logger.error(
            f"Error getting status from port {port} (host {host}) via /internal/status (RequestException): {e_req}"
        )
        status_payload.update(
            {
                "status": STATUS_ERROR,
                "message": f"RequestException while fetching status: {str(e_req)}",
            }
        )
    except Exception as e_gen:  # Catch any other unexpected error
        logger.error(
            f"Unexpected error getting status from port {port} (host {host}): {e_gen}",
            exc_info=True,
        )
        status_payload.update(
            {
                "status": STATUS_ERROR,
                "message": f"Unexpected error fetching status: {str(e_gen)}",
            }
        )
    return status_payload


def start_instance(
    port: int,
    mjpeg_url: str,
    server_config_path: str,
    host: str = "127.0.0.1",
    log_level: str = "info",
) -> Dict[str, Any]:
    """
    Start a server instance on the specified port.
    """
    if is_port_in_use(port, host):
        current_status_info = get_instance_status(port, host)
        # If it's our server and running, or even unresponsive but ours, consider it "already_running" or "unresponsive_ours"
        if current_status_info["status"] in [
            STATUS_RUNNING,
            STATUS_UNRESPONSIVE,
            STATUS_ERROR,
        ]:
            # Further check if it's likely "our" server vs a random process
            # For now, if port is in use and get_instance_status returns one of these, assume it might be ours.
            logger.info(
                f"Server may already be active or port {port} (host {host}) in a conflicting state: {current_status_info['status']}"
            )
            return {
                "port": port,
                "host": host,
                "status": "already_active_or_conflicting",
                "details": current_status_info,
            }
        else:  # e.g. STATUS_STOPPED but port is in use (should not happen if is_port_in_use is accurate)
            logger.warning(
                f"Port {port} (host {host}) is in use by another process or in an unknown state."
            )
            return {
                "port": port,
                "host": host,
                "status": "port_in_use_by_other",
                "message": "Port is in use, not by a recognizable server instance.",
            }

    logger.info(
        f"Starting server instance on {host}:{port} with MJPEG URL: {mjpeg_url}"
    )
    cmd = [
        sys.executable,  # Use sys.executable for python interpreter
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

    process: Optional[subprocess.Popen] = None
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        log_file_name = SERVER_LOG_FILENAME_FORMAT.format(
            port=port
        )  # Use constant format
        log_file_path = os.path.join(LOGS_DIR, log_file_name)

        with open(log_file_path, "a", encoding="utf-8", buffering=1) as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout (goes to same log file)
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP
                    if sys.platform == "win32"
                    else 0
                ),
            )
            logger.info(
                f"Started process for {host}:{port} with PID {process.pid}. Logs: {log_file_path}"
            )

            start_time = time.monotonic()
            while time.monotonic() - start_time < PROCESS_START_TIMEOUT_SECONDS:
                if process.poll() is not None:  # Process terminated prematurely
                    exit_code = process.poll()
                    logger.error(
                        f"Server process {process.pid} ({host}:{port}) terminated prematurely with exit code {exit_code}."
                    )
                    return {
                        "port": port,
                        "host": host,
                        "status": "failed_process_terminated",
                        "pid": process.pid,
                        "exit_code": exit_code,
                        "message": f"Process terminated unexpectedly with exit code {exit_code}.",
                    }

                if is_port_in_use(port, host):
                    time.sleep(1.0)  # Give a moment for API to come up
                    status_check_result = get_instance_status(port, host)
                    current_api_status = status_check_result.get("status")

                    if current_api_status == STATUS_RUNNING:
                        logger.info(
                            f"Server instance on {host}:{port} started successfully (API reports running)."
                        )
                        return {
                            "port": port,
                            "host": host,
                            "status": "started_running",
                            "pid": process.pid,
                            "details": status_check_result.get("details", {}),
                        }

                    # Be more tolerant of transient "error" or "unresponsive" if process is alive
                    if current_api_status in [STATUS_UNRESPONSIVE, STATUS_ERROR]:
                        logger.info(
                            f"Server on {host}:{port} (PID {process.pid}): port in use, API status '{current_api_status}'. Message: '{status_check_result.get('message', 'N/A')}'. Will continue to wait for stable state."
                        )
                        # Continue waiting, do not terminate yet if process is alive.
                    else:  # Other statuses, e.g. if get_instance_status returned something unexpected while port is in use
                        logger.warning(
                            f"Server on {host}:{port} (PID {process.pid}): port in use, API status '{current_api_status}'. Waiting. Message: '{status_check_result.get('message', 'N/A')}'"
                        )

                time.sleep(0.5)  # Polling interval

        # Loop finished: Timeout
        logger.error(
            f"Server instance on {host}:{port} (PID {process.pid if process else 'N/A'}) failed to report 'running' API status within {PROCESS_START_TIMEOUT_SECONDS}s (Timeout)."
        )
        final_status_after_timeout = get_instance_status(port, host)  # One last check

        if process and process.poll() is None:  # If process still alive after timeout
            logger.warning(
                f"Terminating process {process.pid} for {host}:{port} due to startup timeout."
            )
            try:
                process.terminate()
                process.wait(timeout=5.0)  # Wait for termination
                if process.poll() is None:
                    logger.warning(
                        f"Process {process.pid} did not terminate gracefully, killing."
                    )
                    process.kill()
                    process.wait(timeout=2.0)
            except Exception as e_term:
                logger.error(
                    f"Error terminating process {process.pid} for {host}:{port} after timeout: {e_term}"
                )

        return {
            "port": port,
            "host": host,
            "status": "failed_startup_timeout",
            "pid": process.pid if process else None,
            "message": f"Server failed to start and report running API status within timeout. Final API status: {final_status_after_timeout.get('status', 'unknown')}",
        }

    except FileNotFoundError as e_fnf:  # e.g. main_instance.py not found
        logger.error(
            f"Error starting server instance on {host}:{port}: Required file not found - {e_fnf}",
            exc_info=True,
        )
        return {
            "port": port,
            "host": host,
            "status": STATUS_ERROR,
            "message": f"Failed to start: {str(e_fnf)}",
        }
    except Exception as e:
        logger.exception(f"Unexpected error starting server instance on {host}:{port}")
        if (
            process and process.poll() is None
        ):  # If process was started but an error occurred later in manager
            try:
                process.terminate()
            except:
                pass
        return {
            "port": port,
            "host": host,
            "status": STATUS_ERROR,
            "message": f"Unexpected error: {str(e)}",
        }


def update_mjpeg_url(
    port: int, new_url: str, host: str = "127.0.0.1"
) -> Dict[str, Any]:
    """Update the MJPEG URL of a running server instance."""
    url = f"http://{host}:{port}/internal/update_mjpeg_url"

    status_info = get_instance_status(port, host)
    if status_info["status"] != STATUS_RUNNING:
        logger.error(
            f"Cannot update MJPEG URL: Server on {host}:{port} is not in a running state (current status: {status_info['status']})."
        )
        return {
            "port": port,
            "host": host,
            "status": "error_not_running",
            "message": f"Server not running, status: {status_info['status']}",
        }

    try:
        logger.info(f"Updating MJPEG URL for {host}:{port} to: {new_url}")
        response = requests.put(
            url, json={"new_url": new_url}, timeout=REQUEST_TIMEOUT_SECONDS * 2
        )  # Longer timeout for update

        if response.status_code == 200:
            logger.info(f"Successfully updated MJPEG URL for {host}:{port}.")
            return {
                "port": port,
                "host": host,
                "status": "ok",
                "message": response.json().get(
                    "message", f"MJPEG URL updated to {new_url}"
                ),
            }
        else:
            try:
                error_detail = response.json().get("detail", response.text)
            except json.JSONDecodeError:
                error_detail = response.text
            logger.error(
                f"Failed to update MJPEG URL for {host}:{port}. API Error (HTTP {response.status_code}): {error_detail}"
            )
            return {
                "port": port,
                "host": host,
                "status": STATUS_ERROR,
                "message": f"API error {response.status_code}: {error_detail}",
            }

    except RequestException as e:  # Catches ConnectionError, Timeout, etc.
        logger.error(f"Request failed while updating MJPEG URL for {host}:{port}: {e}")
        return {
            "port": port,
            "host": host,
            "status": STATUS_ERROR,
            "message": f"Request failed: {str(e)}",
        }
    except Exception as e_gen:
        logger.exception(f"Unexpected error updating MJPEG URL for {host}:{port}")
        return {
            "port": port,
            "host": host,
            "status": STATUS_ERROR,
            "message": f"Unexpected error: {str(e_gen)}",
        }


def stop_instance(port: int, host: str = "127.0.0.1") -> Dict[str, Any]:
    """Stop a running server instance."""
    import psutil  # Keep import here as it's specific to this function's advanced stop logic

    logger.info(f"Attempting to stop server instance on {host}:{port}")

    status_info = get_instance_status(port, host)  # Check current status via API first

    # Try graceful shutdown via API first, if it seems to be our server
    if status_info["status"] in [
        STATUS_RUNNING,
        STATUS_UNRESPONSIVE,
    ]:  # Unresponsive might still have API if it comes back
        logger.info(
            f"Server on {host}:{port} appears active or unresponsive, attempting graceful API shutdown..."
        )
        shutdown_url = f"http://{host}:{port}/internal/shutdown"
        try:
            response = requests.post(shutdown_url, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                logger.info(
                    f"Graceful shutdown request sent to {host}:{port}. Waiting for port to free up."
                )
            else:
                logger.warning(
                    f"Graceful shutdown API for {host}:{port} returned HTTP {response.status_code}. Proceeding with process check."
                )
        except RequestException as e_api_shut:
            logger.warning(
                f"Could not stop server on {host}:{port} via API ({shutdown_url}): {e_api_shut}. Will try process termination."
            )

        # Wait for port to become free after API shutdown attempt
        wait_time = 0
        max_wait_api_shutdown = 10.0  # seconds
        while wait_time < max_wait_api_shutdown:
            if not is_port_in_use(port, host):
                logger.info(
                    f"Server on {host}:{port} stopped (port freed after API shutdown request)."
                )
                return {
                    "port": port,
                    "host": host,
                    "status": STATUS_STOPPED,
                    "message": "Server stopped successfully (API initiated).",
                }
            time.sleep(0.5)
            wait_time += 0.5
        logger.warning(
            f"Port {host}:{port} still in use after {max_wait_api_shutdown}s waiting for API shutdown. Proceeding with process termination."
        )

    elif status_info["status"] == STATUS_STOPPED:
        logger.info(
            f"Server on {host}:{port} is already reported as stopped by API check (or port not in use)."
        )
        if is_port_in_use(
            port, host
        ):  # Double check, API might be wrong or it's another process
            logger.warning(
                f"Port {host}:{port} is still in use, though API status was 'stopped'. Will attempt process termination for this port."
            )
        else:
            return {
                "port": port,
                "host": host,
                "status": STATUS_STOPPED,
                "message": "Server already stopped.",
            }
    else:  # Other statuses like error, or if API check itself failed.
        logger.info(
            f"Server on {host}:{port} status is '{status_info['status']}'. Will check for listening process on port."
        )

    # If API shutdown didn't work or wasn't applicable, try to find and terminate the process
    try:
        listening_pid: Optional[int] = None
        for conn in psutil.net_connections(kind="inet"):
            if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
                # Check if conn.laddr.ip matches `host` if host is not '0.0.0.0' or '::'
                # For simplicity, if host is 127.0.0.1, conn.laddr.ip could be '127.0.0.1'.
                # If host is a specific public IP, conn.laddr.ip should match.
                # If host is '0.0.0.0', any local IP is fine.
                # This check can be complex with various network setups.
                # For now, primarily matching by port for local instances.
                if host == "127.0.0.1" and conn.laddr.ip not in [
                    "127.0.0.1",
                    "0.0.0.0",
                    "::",
                ]:
                    continue
                if host not in ["127.0.0.1", "0.0.0.0", "::"] and conn.laddr.ip != host:
                    # If specific host given, and listened IP doesn't match (and isn't wildcard)
                    if conn.laddr.ip not in ["0.0.0.0", "::"]:  # allow wildcard listen
                        continue

                if conn.pid is not None:
                    listening_pid = conn.pid
                    break

        if listening_pid:
            logger.info(
                f"Found process PID {listening_pid} listening on {host}:{port}. Attempting to terminate."
            )
            process = psutil.Process(listening_pid)

            # Check if it's likely our python process (optional, for safety)
            # if "python" not in process.name().lower() and "main_instance.py" not in " ".join(process.cmdline()):
            #     logger.warning(f"Process PID {listening_pid} on port {port} doesn't look like our server. Name: {process.name()}. Skipping termination.")
            #     return {"port": port, "host":host, "status": "error_foreign_process", "message": f"Process PID {listening_pid} on port does not appear to be the target server."}

            process.terminate()  # SIGTERM
            try:
                process.wait(timeout=5.0)
                logger.info(
                    f"Process PID {listening_pid} for {host}:{port} terminated."
                )
                return {
                    "port": port,
                    "host": host,
                    "status": STATUS_STOPPED,
                    "message": f"Process PID {listening_pid} terminated.",
                }
            except psutil.TimeoutExpired:
                logger.warning(
                    f"Process PID {listening_pid} for {host}:{port} did not terminate gracefully. Killing."
                )
                process.kill()  # SIGKILL
                process.wait(timeout=2.0)  # Wait for kill
                logger.info(f"Process PID {listening_pid} for {host}:{port} killed.")
                return {
                    "port": port,
                    "host": host,
                    "status": STATUS_STOPPED,
                    "message": f"Process PID {listening_pid} killed.",
                }
        else:
            if is_port_in_use(port, host):
                logger.warning(
                    f"Port {host}:{port} is in use, but no specific listening process PID found via psutil (or permissions issue)."
                )
                return {
                    "port": port,
                    "host": host,
                    "status": STATUS_ERROR,
                    "message": "Port in use, but failed to identify/terminate process.",
                }
            else:
                logger.info(
                    f"Port {host}:{port} is not in use. Server considered stopped."
                )
                return {
                    "port": port,
                    "host": host,
                    "status": STATUS_STOPPED,
                    "message": "Port not in use.",
                }

    except psutil.NoSuchProcess:
        logger.info(
            f"Process for {host}:{port} already gone before termination attempt."
        )
        return {
            "port": port,
            "host": host,
            "status": STATUS_STOPPED,
            "message": "Process already exited.",
        }
    except psutil.AccessDenied:
        logger.error(
            f"Access denied when trying to terminate process on {host}:{port}."
        )
        return {
            "port": port,
            "host": host,
            "status": STATUS_ERROR,
            "message": "Access denied to terminate process.",
        }
    except Exception as e_term:
        logger.error(
            f"Error during process termination for {host}:{port}: {e_term}",
            exc_info=True,
        )
        return {
            "port": port,
            "host": host,
            "status": STATUS_ERROR,
            "message": f"Error terminating process: {str(e_term)}",
        }


def load_multi_config(config_path: str) -> List[Dict[str, Any]]:
    """Load multi-instance configuration from a JSON file."""
    if not os.path.exists(config_path):
        logger.warning(
            f"Multi-instance configuration file not found: {config_path}. Returning empty list."
        )
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        if not isinstance(config_data, list):
            logger.error(
                f"Invalid multi-config format: {config_path} must contain a JSON list. Found: {type(config_data)}"
            )
            return []

        valid_configs: List[Dict[str, Any]] = []
        for idx, item in enumerate(config_data):
            if not isinstance(item, dict):
                logger.error(
                    f"Invalid item at index {idx} in {config_path}: must be a dictionary. Got: {type(item)}"
                )
                continue
            if "port" not in item or not isinstance(item["port"], int):
                logger.error(
                    f"Missing or invalid 'port' (must be int) in item at index {idx} of {config_path}."
                )
                continue
            if "mjpeg_url" not in item or not isinstance(item["mjpeg_url"], str):
                logger.error(
                    f"Missing or invalid 'mjpeg_url' (must be str) in item at index {idx} of {config_path}."
                )
                continue
            # Add optional host, default to 127.0.0.1 if not present
            item.setdefault("host", "127.0.0.1")
            if not isinstance(item["host"], str):
                logger.error(
                    f"Invalid 'host' (must be str) in item at index {idx} of {config_path}. Using default '127.0.0.1'."
                )
                item["host"] = "127.0.0.1"

            valid_configs.append(item)

        logger.info(
            f"Loaded {len(valid_configs)} valid instance configurations from {config_path}"
        )
        return valid_configs
    except json.JSONDecodeError as e_json:
        logger.error(f"Error parsing JSON in multi-config file {config_path}: {e_json}")
        return []
    except Exception:
        logger.exception(f"Unexpected error loading multi-config from {config_path}")
        return []


def save_multi_config(config_path: str, instances: List[Dict[str, Any]]) -> bool:
    """Create or update multi-instance configuration file."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2)
        logger.info(f"Saved {len(instances)} instance configurations to {config_path}")
        return True
    except Exception:
        logger.exception(f"Error saving multi-config to {config_path}")
        return False


def main():
    """Main entry point for the server manager."""
    parser = argparse.ArgumentParser(
        description="Manage multiple instances of the Image Matching API Server",
        formatter_class=argparse.RawTextHelpFormatter,  # Allows for better help text formatting
    )

    parser.add_argument(
        "--multi-config",
        default=DEFAULT_MANAGER_MULTI_CONFIG_PATH,
        help=f"Path to multi-instance JSON configuration file (default: {DEFAULT_MANAGER_MULTI_CONFIG_PATH})",
    )
    parser.add_argument(
        "--server-config",
        default=DEFAULT_SERVER_CONFIG_PATH,
        help=f"Path to individual server's INI configuration file (default: {DEFAULT_SERVER_CONFIG_PATH})",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level for the manager (default: info)",
    )

    subparsers = parser.add_subparsers(dest="command", title="Commands", required=True)

    # Start command
    start_p = subparsers.add_parser(
        "start",
        help="Start server instances defined in multi-config, or a specific one.",
    )
    start_p.add_argument(
        "--port",
        type=int,
        help="Port of a specific instance to start. If not in multi-config, --mjpeg-url and optionally --host must be provided.",
    )
    start_p.add_argument(
        "--mjpeg-url",
        help="MJPEG URL for the specific instance (if --port is given and not in config).",
    )
    start_p.add_argument(
        "--host",
        help="Host for the specific instance (if --port is given). Defaults to 127.0.0.1 or config value.",
    )

    # Stop command
    stop_p = subparsers.add_parser(
        "stop", help="Stop server instances from multi-config, or a specific one."
    )
    stop_p.add_argument("--port", type=int, help="Port of a specific instance to stop.")
    stop_p.add_argument(
        "--host",
        help="Host of the specific instance to stop (defaults to 127.0.0.1 or config value).",
    )

    # Status command
    status_p = subparsers.add_parser(
        "status", help="Get status of instances from multi-config, or a specific one."
    )
    status_p.add_argument(
        "--port", type=int, help="Port of a specific instance to get status for."
    )
    status_p.add_argument(
        "--host",
        help="Host of the specific instance (defaults to 127.0.0.1 or config value).",
    )

    # Update MJPEG URL command
    update_p = subparsers.add_parser(
        "update-mjpeg", help="Update MJPEG URL for a running instance."
    )
    update_p.add_argument(
        "--port", type=int, required=True, help="Port of the instance to update."
    )
    update_p.add_argument(
        "--host", default="127.0.0.1", help="Host of the instance (default: 127.0.0.1)."
    )
    update_p.add_argument("--new-mjpeg-url", required=True, help="New MJPEG URL.")

    # Config management commands
    config_p = subparsers.add_parser(
        "config", help="Manage multi-instance configurations."
    )
    config_sp = config_p.add_subparsers(
        dest="config_command", title="Config Subcommands", required=True
    )

    # Config Add
    cfg_add_p = config_sp.add_parser(
        "add", help="Add a new server instance to the multi-config."
    )
    cfg_add_p.add_argument(
        "--port", type=int, required=True, help="Port for the new instance."
    )
    cfg_add_p.add_argument(
        "--mjpeg-url", required=True, help="MJPEG URL for the new instance."
    )
    cfg_add_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the new instance (default: 127.0.0.1).",
    )

    # Config Remove
    cfg_remove_p = config_sp.add_parser(
        "remove", help="Remove an instance from the multi-config."
    )
    cfg_remove_p.add_argument(
        "--port", type=int, required=True, help="Port of the instance to remove."
    )
    cfg_remove_p.add_argument(
        "--host",
        help="Host of the instance to remove (optional, used for matching if provided).",
    )

    # Config List
    config_sp.add_parser("list", help="List all instances in the multi-config.")

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    multi_config_instances = load_multi_config(args.multi_config)

    results_output = []  # For commands that produce multiple results

    if args.command == "start":
        if args.port:
            instance_to_start = next(
                (
                    inst
                    for inst in multi_config_instances
                    if inst["port"] == args.port
                    and inst.get("host", "127.0.0.1")
                    == (args.host or inst.get("host", "127.0.0.1"))
                ),
                None,
            )
            mjpeg_url_to_use = args.mjpeg_url
            host_to_use = args.host or "127.0.0.1"

            if instance_to_start:  # Found in config
                mjpeg_url_to_use = (
                    args.mjpeg_url or instance_to_start["mjpeg_url"]
                )  # CLI override or config
                host_to_use = args.host or instance_to_start.get("host", "127.0.0.1")
            elif not args.mjpeg_url:  # Not in config AND no mjpeg_url provided via CLI
                logger.error(
                    f"Instance for port {args.port} (host {host_to_use}) not in multi-config and --mjpeg-url not provided."
                )
                return 1

            logger.info(
                f"Attempting to start instance: Port={args.port}, Host={host_to_use}, MJPEG_URL={mjpeg_url_to_use}"
            )
            result = start_instance(
                args.port,
                mjpeg_url_to_use,
                args.server_config,
                host_to_use,
                args.log_level,
            )
            results_output.append(result)
        else:  # Start all from multi-config
            if not multi_config_instances:
                logger.info("No instances in multi-config to start.")
            for inst_cfg in multi_config_instances:
                logger.info(
                    f"Attempting to start instance from config: Port={inst_cfg['port']}, Host={inst_cfg.get('host', '127.0.0.1')}, MJPEG_URL={inst_cfg['mjpeg_url']}"
                )
                result = start_instance(
                    inst_cfg["port"],
                    inst_cfg["mjpeg_url"],
                    args.server_config,
                    inst_cfg.get("host", "127.0.0.1"),
                    args.log_level,
                )
                results_output.append(result)

    elif args.command == "stop":
        if args.port:
            host_to_use = args.host
            if (
                not host_to_use
            ):  # If host not given, try to find it from config for this port
                cfg_item = next(
                    (c for c in multi_config_instances if c["port"] == args.port), None
                )
                host_to_use = (
                    cfg_item.get("host", "127.0.0.1") if cfg_item else "127.0.0.1"
                )
            results_output.append(stop_instance(args.port, host_to_use))
        else:  # Stop all from multi-config
            if not multi_config_instances:
                logger.info("No instances in multi-config to stop.")
            for inst_cfg in multi_config_instances:
                results_output.append(
                    stop_instance(inst_cfg["port"], inst_cfg.get("host", "127.0.0.1"))
                )

    elif args.command == "status":
        if args.port:
            host_to_use = args.host
            if not host_to_use:
                cfg_item = next(
                    (c for c in multi_config_instances if c["port"] == args.port), None
                )
                host_to_use = (
                    cfg_item.get("host", "127.0.0.1") if cfg_item else "127.0.0.1"
                )
            results_output.append(get_instance_status(args.port, host_to_use))
        else:  # Status all from multi-config
            if not multi_config_instances:
                logger.info("No instances in multi-config for status check.")
            for inst_cfg in multi_config_instances:
                results_output.append(
                    get_instance_status(
                        inst_cfg["port"], inst_cfg.get("host", "127.0.0.1")
                    )
                )

    elif args.command == "update-mjpeg":
        result = update_mjpeg_url(args.port, args.new_mjpeg_url, args.host)
        results_output.append(result)
        if (
            result.get("status") == "ok"
        ):  # If API update was successful, update multi-config
            updated = False
            for inst_cfg in multi_config_instances:
                if (
                    inst_cfg["port"] == args.port
                    and inst_cfg.get("host", "127.0.0.1") == args.host
                ):
                    inst_cfg["mjpeg_url"] = args.new_mjpeg_url
                    updated = True
                    break
            if updated:
                save_multi_config(args.multi_config, multi_config_instances)
            else:
                logger.warning(
                    f"MJPEG URL updated for running instance {args.host}:{args.port}, but this instance was not found in multi-config {args.multi_config} to save the change."
                )

    elif args.command == "config":
        if args.config_command == "add":
            # Check if port/host combo already exists
            if any(
                inst["port"] == args.port and inst.get("host", "127.0.0.1") == args.host
                for inst in multi_config_instances
            ):
                logger.error(
                    f"Instance {args.host}:{args.port} already exists in {args.multi_config}."
                )
                return 1
            multi_config_instances.append(
                {"port": args.port, "mjpeg_url": args.mjpeg_url, "host": args.host}
            )
            if save_multi_config(args.multi_config, multi_config_instances):
                results_output.append(
                    {
                        "status": "ok",
                        "message": f"Instance {args.host}:{args.port} added to {args.multi_config}.",
                    }
                )
            else:
                results_output.append(
                    {
                        "status": STATUS_ERROR,
                        "message": f"Failed to save update to {args.multi_config}.",
                    }
                )

        elif args.config_command == "remove":
            initial_len = len(multi_config_instances)
            host_to_match = args.host  # Can be None

            multi_config_instances = [
                inst
                for inst in multi_config_instances
                if not (
                    inst["port"] == args.port
                    and (
                        host_to_match is None
                        or inst.get("host", "127.0.0.1") == host_to_match
                    )
                )
            ]
            if len(multi_config_instances) < initial_len:
                if save_multi_config(args.multi_config, multi_config_instances):
                    results_output.append(
                        {
                            "status": "ok",
                            "message": f"Instance(s) for port {args.port} (host: {host_to_match or 'any'}) removed from {args.multi_config}.",
                        }
                    )
                else:
                    results_output.append(
                        {
                            "status": STATUS_ERROR,
                            "message": f"Failed to save update to {args.multi_config} after removal.",
                        }
                    )
            else:
                results_output.append(
                    {
                        "status": "not_found",
                        "message": f"No instance for port {args.port} (host: {host_to_match or 'any'}) found in {args.multi_config}.",
                    }
                )

        elif args.config_command == "list":
            if not multi_config_instances:
                results_output.append(
                    {"message": f"No instances configured in {args.multi_config}."}
                )
            else:
                results_output = multi_config_instances  # List the instances themselves

    # Print results if any
    if results_output:
        print(json.dumps(results_output, indent=2))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Manager interrupted by user (KeyboardInterrupt).")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e_main:  # Catch-all for unexpected errors in main logic
        logger.critical(
            f"Manager encountered a critical unexpected error: {e_main}", exc_info=True
        )
        sys.exit(1)
