from prometheus_client import Counter, Histogram


HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests.",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "endpoint"],
)

ML_PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total number of ML predictions produced.",
    ["model_version"],
)

ML_DECISIONS_TOTAL = Counter(
    "ml_decisions_total",
    "Total number of business decisions produced.",
    ["policy_version", "decision"],
)