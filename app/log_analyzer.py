"""
Log analyzer component for AIOps assistant.
Analyzes log lines and extracts structured insights.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR    = "ERROR"
    WARNING  = "WARNING"
    INFO     = "INFO"
    DEBUG    = "DEBUG"
    UNKNOWN  = "UNKNOWN"


@dataclass
class LogEntry:
    """Structured representation of a parsed log line."""
    raw:        str
    severity:   Severity
    message:    str
    timestamp:  Optional[str]
    service:    Optional[str]
    error_code: Optional[str]


# Regex patterns for common log formats
SEVERITY_PATTERNS = {
    Severity.CRITICAL: re.compile(
        r"\b(CRITICAL|FATAL|EMERG)\b", re.IGNORECASE
    ),
    Severity.ERROR: re.compile(
        r"\b(ERROR|ERR|EXCEPTION|TRACEBACK)\b", re.IGNORECASE
    ),
    Severity.WARNING: re.compile(
        r"\b(WARNING|WARN)\b", re.IGNORECASE
    ),
    Severity.INFO: re.compile(
        r"\b(INFO|INFORMATION)\b", re.IGNORECASE
    ),
    Severity.DEBUG: re.compile(
        r"\b(DEBUG|TRACE|VERBOSE)\b", re.IGNORECASE
    ),
}

TIMESTAMP_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)

ERROR_CODE_PATTERN = re.compile(
    r"\b([A-Z]{2,6}[-_]?\d{3,6})\b"
)

HTTP_STATUS_PATTERN = re.compile(
    r"\b([45]\d{2})\b"
)


class LogAnalyzer:
    """
    Analyzes log lines to extract severity, patterns, and insights.
    """

    def parse_line(self, log_line: str) -> LogEntry:
        """
        Parse a single log line into a structured LogEntry.

        Args:
            log_line: Raw log line string

        Returns:
            LogEntry with extracted fields
        """
        if not log_line or not log_line.strip():
            return LogEntry(
                raw=log_line,
                severity=Severity.UNKNOWN,
                message="",
                timestamp=None,
                service=None,
                error_code=None
            )

        severity   = self._detect_severity(log_line)
        timestamp  = self._extract_timestamp(log_line)
        error_code = self._extract_error_code(log_line)

        return LogEntry(
            raw=log_line,
            severity=severity,
            message=log_line.strip(),
            timestamp=timestamp,
            service=None,
            error_code=error_code
        )

    def analyze_batch(self, log_lines: List[str]) -> Dict:
        """
        Analyze a batch of log lines and return summary statistics.

        Args:
            log_lines: List of raw log line strings

        Returns:
            Dictionary with counts, severity breakdown, and top errors
        """
        entries = [self.parse_line(line) for line in log_lines]

        severity_counts = {s.value: 0 for s in Severity}
        for entry in entries:
            severity_counts[entry.severity.value] += 1

        error_entries = [
            e for e in entries
            if e.severity in (Severity.ERROR, Severity.CRITICAL)
        ]

        return {
            "total_lines":        len(entries),
            "severity_breakdown": severity_counts,
            "error_count":        len(error_entries),
            "has_critical":       severity_counts[Severity.CRITICAL.value] > 0,
            "top_errors":         [e.message[:200] for e in error_entries[:5]],
        }

    def _detect_severity(self, log_line: str) -> Severity:
        """Detect the severity level from a log line."""
        for severity, pattern in SEVERITY_PATTERNS.items():
            if pattern.search(log_line):
                return severity

        http_match = HTTP_STATUS_PATTERN.search(log_line)
        if http_match:
            code = int(http_match.group(1))
            if code >= 500:
                return Severity.ERROR
            elif code >= 400:
                return Severity.WARNING

        return Severity.UNKNOWN

    def _extract_timestamp(self, log_line: str) -> Optional[str]:
        """Extract ISO-style timestamp from log line."""
        match = TIMESTAMP_PATTERN.search(log_line)
        return match.group(1) if match else None

    def _extract_error_code(self, log_line: str) -> Optional[str]:
        """Extract error codes like OOM-001, K8S-404 etc."""
        match = ERROR_CODE_PATTERN.search(log_line)
        return match.group(1) if match else None
