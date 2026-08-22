"""
AIOps Assistant — Main FastAPI Application
Provides a REST API for operational intent classification
and log analysis.
"""

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.intent_classifier import get_classifier
from app.log_analyzer import LogAnalyzer
from app.health import get_liveness, get_readiness

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AIOps Assistant",
    description="AI-powered operations assistant for Kubernetes",
    version="1.0.0"
)

# Initialize components
log_analyzer = LogAnalyzer()


# ── Request / Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:   str
    context: Optional[str] = None


class QueryResponse(BaseModel):
    intent:           str
    confidence:       float
    query:            str
    suggested_action: str


class LogAnalysisRequest(BaseModel):
    logs: List[str]


class LogAnalysisResponse(BaseModel):
    total_lines:        int
    error_count:        int
    has_critical:       bool
    severity_breakdown: dict
    top_errors:         List[str]


# ── Intent Action Suggestions ─────────────────────────────────────────────────

INTENT_ACTIONS = {
    "metrics":     "Query Prometheus or run: kubectl top pods",
    "remediation": "Review runbook and execute: kubectl rollout restart",
    "logs":        "Run: kubectl logs <pod-name> --tail=100",
    "health":      "Run: kubectl get pods && kubectl describe pod <name>",
    "unknown":     "Please rephrase your query with more operational context",
}


# ── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing information."""
    start    = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration}ms"
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "AIOps Assistant",
        "version": "1.0.0",
        "status":  "running",
        "docs":    "/docs"
    }


@app.get("/health/live")
async def liveness():
    """Kubernetes liveness probe endpoint."""
    return get_liveness()


@app.get("/health/ready")
async def readiness():
    """Kubernetes readiness probe endpoint."""
    try:
        return get_readiness()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def classify_query(request: QueryRequest):
    """
    Classify a natural language operational query.

    Example:
        POST /query
        {"query": "show me cpu usage for the payment service"}
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:
        classifier      = get_classifier()
        intent, confidence = classifier.predict(request.query)

        logger.info(
            f"Query classified: intent={intent} "
            f"confidence={confidence:.3f} "
            f"query='{request.query[:50]}'"
        )

        return QueryResponse(
            intent=intent,
            confidence=round(confidence, 4),
            query=request.query,
            suggested_action=INTENT_ACTIONS.get(
                intent, "No action available"
            )
        )

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Classification failed"
        )


@app.post("/analyze/logs", response_model=LogAnalysisResponse)
async def analyze_logs(request: LogAnalysisRequest):
    """
    Analyze a batch of log lines and return structured insights.

    Example:
        POST /analyze/logs
        {"logs": ["2024-01-01 ERROR: OOM killed", "INFO: Pod started"]}
    """
    if not request.logs:
        raise HTTPException(
            status_code=400,
            detail="Logs list cannot be empty"
        )

    try:
        result = log_analyzer.analyze_batch(request.logs)
        return LogAnalysisResponse(**result)

    except Exception as e:
        logger.error(f"Log analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Log analysis failed"
        )


@app.get("/metrics/info")
async def metrics_info():
    """Return basic service metrics information."""
    return {
        "service":            "aiops-assistant",
        "classifier_status":  "trained",
        "supported_intents":  list(INTENT_ACTIONS.keys()),
    }
