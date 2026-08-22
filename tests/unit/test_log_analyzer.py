"""
Unit tests for the LogAnalyzer component.
Tests parsing, severity detection, and batch analysis.
"""

import pytest
from app.log_analyzer import LogAnalyzer, LogEntry, Severity


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    """Fresh LogAnalyzer for each test."""
    return LogAnalyzer()


# ── Sample Log Lines ──────────────────────────────────────────────────────────

SAMPLE_LOGS = {
    "error":       "2024-01-15 10:23:45 ERROR: Database connection failed",
    "critical":    "2024-01-15 10:23:46 CRITICAL: OOM Killer activated",
    "warning":     "2024-01-15 10:23:47 WARNING: High memory usage at 85%",
    "info":        "2024-01-15 10:23:48 INFO: Pod started successfully",
    "debug":       "2024-01-15 10:23:49 DEBUG: Processing request ID 12345",
    "http_500":    "POST /api/query HTTP/1.1 500 Internal Server Error",
    "http_404":    "GET /health HTTP/1.1 404 Not Found",
    "no_severity": "some random text without severity marker",
}


# ── Parsing Tests ─────────────────────────────────────────────────────────────

class TestLogParsing:

    def test_parse_returns_log_entry(self, analyzer):
        """parse_line() should return a LogEntry object."""
        result = analyzer.parse_line(SAMPLE_LOGS["error"])
        assert isinstance(result, LogEntry)

    def test_parse_empty_string(self, analyzer):
        """Empty string should return UNKNOWN severity."""
        entry = analyzer.parse_line("")
        assert entry.severity == Severity.UNKNOWN
        assert entry.message == ""

    def test_parse_preserves_raw_line(self, analyzer):
        """raw field should contain the original unmodified line."""
        raw   = SAMPLE_LOGS["error"]
        entry = analyzer.parse_line(raw)
        assert entry.raw == raw

    def test_parse_extracts_timestamp(self, analyzer):
        """Timestamp should be extracted from ISO-format logs."""
        entry = analyzer.parse_line(SAMPLE_LOGS["error"])
        assert entry.timestamp is not None
        assert "2024-01-15" in entry.timestamp


# ── Severity Detection Tests ──────────────────────────────────────────────────

class TestSeverityDetection:

    @pytest.mark.parametrize("log_key,expected_severity", [
        ("error",       Severity.ERROR),
        ("critical",    Severity.CRITICAL),
        ("warning",     Severity.WARNING),
        ("info",        Severity.INFO),
        ("debug",       Severity.DEBUG),
        ("http_500",    Severity.ERROR),
        ("http_404",    Severity.WARNING),
        ("no_severity", Severity.UNKNOWN),
    ])
    def test_severity_detection(
        self, analyzer, log_key, expected_severity
    ):
        """Each log type should be detected with correct severity."""
        entry = analyzer.parse_line(SAMPLE_LOGS[log_key])
        assert entry.severity == expected_severity, (
            f"Log '{log_key}' detected as {entry.severity}, "
            f"expected {expected_severity}"
        )

    def test_case_insensitive_severity(self, analyzer):
        """Severity detection should be case-insensitive."""
        lower = analyzer.parse_line("error: something failed")
        upper = analyzer.parse_line("ERROR: something failed")
        mixed = analyzer.parse_line("Error: something failed")
        assert lower.severity == Severity.ERROR
        assert upper.severity == Severity.ERROR
        assert mixed.severity == Severity.ERROR

    def test_critical_takes_priority_over_error(self, analyzer):
        """CRITICAL should be detected even when ERROR also appears."""
        entry = analyzer.parse_line(
            "CRITICAL: Multiple ERROR conditions detected"
        )
        assert entry.severity == Severity.CRITICAL


# ── Batch Analysis Tests ──────────────────────────────────────────────────────

class TestBatchAnalysis:

    def test_analyze_batch_returns_dict(self, analyzer):
        """analyze_batch() should return a dictionary."""
        result = analyzer.analyze_batch(list(SAMPLE_LOGS.values()))
        assert isinstance(result, dict)

    def test_analyze_batch_total_lines_count(self, analyzer):
        """total_lines should equal the number of input lines."""
        lines  = list(SAMPLE_LOGS.values())
        result = analyzer.analyze_batch(lines)
        assert result["total_lines"] == len(lines)

    def test_analyze_batch_counts_errors(self, analyzer):
        """error_count should include ERROR and CRITICAL lines."""
        logs = [
            "ERROR: first error",
            "CRITICAL: critical issue",
            "INFO: normal operation",
            "WARNING: watch out",
        ]
        result = analyzer.analyze_batch(logs)
        assert result["error_count"] == 2

    def test_analyze_batch_detects_critical(self, analyzer):
        """has_critical flag should be True when CRITICAL exists."""
        logs = [
            "INFO: normal",
            "CRITICAL: system failure",
            "ERROR: database down",
        ]
        result = analyzer.analyze_batch(logs)
        assert result["has_critical"] is True

    def test_analyze_batch_no_critical(self, analyzer):
        """has_critical should be False when no CRITICAL lines exist."""
        logs = [
            "INFO: started",
            "ERROR: minor issue",
            "WARNING: check this"
        ]
        result = analyzer.analyze_batch(logs)
        assert result["has_critical"] is False

    def test_analyze_batch_top_errors_limited_to_5(self, analyzer):
        """top_errors should never return more than 5 entries."""
        error_logs = [f"ERROR: error number {i}" for i in range(20)]
        result     = analyzer.analyze_batch(error_logs)
        assert len(result["top_errors"]) <= 5

    def test_analyze_empty_batch(self, analyzer):
        """Empty list should return zero counts."""
        result = analyzer.analyze_batch([])
        assert result["total_lines"] == 0
        assert result["error_count"] == 0
        assert result["has_critical"] is False
