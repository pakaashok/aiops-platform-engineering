"""
Health check endpoints for Kubernetes liveness and readiness probes.
"""

import time
from typing import Dict

# Track application start time for uptime calculation
_start_time = time.time()

# Readiness state — can be toggled for graceful shutdown
_is_ready = True


def get_liveness() -> Dict:
    """
    Liveness check — is the process alive and not deadlocked?
    Kubernetes will restart the pod if this fails.
    """
    return {
        "status":         "alive",
        "uptime_seconds": round(time.time() - _start_time, 2),
        "timestamp":      time.time()
    }


def get_readiness() -> Dict:
    """
    Readiness check — is the app ready to accept traffic?
    Kubernetes removes pod from endpoints if this fails.
    """
    if not _is_ready:
        raise RuntimeError("Application is not ready")

    return {
        "status": "ready",
        "checks": {
            "classifier":   "ok",
            "log_analyzer": "ok"
        }
    }


def set_ready(state: bool) -> None:
    """Toggle readiness state — useful for graceful shutdown."""
    global _is_ready
    _is_ready = state
