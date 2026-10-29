"""Efficio Apex: AI-Powered Smart Ventilation System.
Primary Application Entry Point.

Architecture Relationship:
- control process (control/main.py): Reads sensor data, performs baseline tracking and ML inference,
  updates the state machine, and drives the fan relay.
- server process (server/app.py): Read-only FastAPI dashboard API and WebSocket server.
- root entry point (main.py): Unified launcher allowing single-command startup of both services
  or independent execution of either service, and exports the FastAPI 'app' instance.

Usage:
  python main.py                 # Runs both control and web server in simulation/configured mode
  python main.py --mode control  # Runs only the control loop (equivalent to python -m control.main)
  python main.py --mode server   # Runs only the web dashboard (equivalent to uvicorn server.app:app)
  uvicorn main:app --port 8000   # Direct ASGI runner
"""

import argparse
import sys
import threading
import time
import uvicorn

from common import config
from common.logger import get_logger, setup_logging
from server.app import app  # Exported FastAPI instance for ASGI servers

logger = get_logger("main")


def run_control():
    """Run the control loop."""
    from control.main import main as control_main
    control_main()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI web dashboard server."""
    uvicorn.run(app, host=host, port=port, log_level="info")


def main():
    parser = argparse.ArgumentParser(
        description="Efficio Apex: AI-Powered Smart Ventilation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "control", "server"],
        default="all",
        help="Service execution mode (default: all)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for dashboard server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for dashboard server (default: 8000)",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Optional path to custom config.yaml",
    )
    args = parser.parse_args()

    # Load and validate configuration
    try:
        cfg = config.load(args.config_path)
        setup_logging(
            level=cfg.get("logging", {}).get("level", "INFO"),
            format_str=cfg.get("logging", {}).get("format"),
        )
    except Exception as e:
        print(f"[FATAL] Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    logger.info(
        "Initializing Efficio Apex [source: %s, mode: %s, hardware_enabled: %s, fan_control_enabled: %s]",
        cfg.get("source"),
        args.mode,
        cfg.get("hardware_enabled", False),
        cfg.get("fan_control_enabled", False),
    )

    if args.mode == "control":
        run_control()
    elif args.mode == "server":
        run_server(host=args.host, port=args.port)
    elif args.mode == "all":
        # Launch control loop in background daemon thread
        control_thread = threading.Thread(target=run_control, daemon=True, name="ControlThread")
        control_thread.start()
        logger.info("Control loop started in background thread. Launching web dashboard on http://%s:%d", args.host, args.port)

        try:
            run_server(host=args.host, port=args.port)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutting down application...")


if __name__ == "__main__":
    main()
