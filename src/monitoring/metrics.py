"""The service's health in numbers, for Prometheus.

Prometheus visits GET /metrics every few seconds (monitoring/prometheus/
prometheus.yml), reads the numbers below, and stores each with the time.

Three kinds of number:
  Counter    only goes up, like a car's odometer. Prometheus turns "how fast
             it grows" into a rate: requests per second, refusals per minute.
  Histogram  tally marks in time buckets (<= 5 ms, <= 10 ms, ... <= 10 s), from
             which Prometheus works out "95% of requests took at most X".
  Gauge      goes up and down, like a thermometer.

Requests are labelled by ROUTE (e.g. "/predict"), never by the raw address:
a thousand random URLs must not become a thousand separate counters. Any
address that matches no route counts as "unmatched".
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

__all__ = ["CONTENT_TYPE_LATEST", "ServiceMetrics"]


class ServiceMetrics:
    """One service's numbers, in a registry of their own.

    Each app gets its own registry (not prometheus_client's global one), so
    the tests -- which build many apps -- each start from zero.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "rul_http_requests_total", "HTTP requests answered, by route and status code.",
            ["method", "route", "status"], registry=self.registry,
        )
        self.duration = Histogram(
            "rul_http_request_duration_seconds", "Time to answer an HTTP request, in seconds.",
            ["route"], registry=self.registry,
        )
        self.warnings = Counter(
            "rul_predictions_with_warnings_total",
            "Predictions answered with at least one warning (a reading outside the training range).",
            registry=self.registry,
        )
        self.diary_failures = Counter(
            "rul_prediction_log_failures_total", "Predictions that could not be written to the diary.",
            registry=self.registry,
        )
        self.model = Gauge(
            "rul_model_info", "The model answering requests (the value is always 1; read the labels).",
            ["model_name", "model_version"], registry=self.registry,
        )

    def page(self) -> bytes:
        """The /metrics page: every number above, in Prometheus's text format."""
        return generate_latest(self.registry)
