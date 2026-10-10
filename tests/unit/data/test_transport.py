"""Transport: pacing, backoff, and the seam that makes parsers testable."""

from __future__ import annotations

import time

import pytest

from tradingview_mcp.core.data.transport import (
    FakeTransport,
    HostRateLimiter,
    HttpxTransport,
    RetryPolicy,
    TokenBucket,
    TransportError,
    TransportResponse,
    json_response,
)


class TestFakeTransport:
    def test_routes_by_substring(self) -> None:
        tx = FakeTransport().add_text("egx.com.eg", "<html>ok</html>")
        assert tx.request("GET", "https://www.egx.com.eg/en/StocksData.aspx").text == "<html>ok</html>"

    def test_unmatched_route_raises_non_transient(self) -> None:
        tx = FakeTransport()
        with pytest.raises(TransportError) as exc:
            tx.request("GET", "https://example.com/x")
        assert exc.value.transient is False

    def test_exception_route_is_raised(self) -> None:
        tx = FakeTransport().add("yahoo", TransportError("boom", transient=True))
        with pytest.raises(TransportError):
            tx.request("GET", "https://query1.finance.yahoo.com/x")

    def test_records_calls_for_assertions(self) -> None:
        tx = FakeTransport().add_text("egx", "x")
        tx.request("GET", "https://egx.com.eg/a")
        tx.request("GET", "https://egx.com.eg/b")
        assert tx.route_count("egx.com.eg") == 2
        assert tx.called_with("/b")

    def test_json_response_helper(self) -> None:
        resp = json_response({"a": 1})
        assert resp.json() == {"a": 1}


class TestTokenBucket:
    def test_burst_consumes_immediately(self) -> None:
        bucket = TokenBucket(rate_per_s=1000, burst=3)
        assert bucket.acquire(3) == 0.0

    def test_empty_bucket_waits(self) -> None:
        bucket = TokenBucket(rate_per_s=20, burst=1)
        bucket.acquire(1)
        started = time.monotonic()
        bucket.acquire(1)
        assert time.monotonic() - started >= 0.02

    def test_rejects_non_positive_rate(self) -> None:
        with pytest.raises(ValueError):
            TokenBucket(rate_per_s=0)


class TestHostRateLimiter:
    def test_per_host_overrides(self) -> None:
        limiter = HostRateLimiter(default_rate=1000, default_burst=1000)
        limiter.set_host_rate("www.egx.com.eg", 1000, 1)
        assert limiter.acquire("https://www.egx.com.eg/a") == 0.0
        started = time.monotonic()
        limiter.acquire("https://www.egx.com.eg/b")
        assert time.monotonic() - started >= 0.0005

    def test_hosts_do_not_starve_each_other(self) -> None:
        limiter = HostRateLimiter(default_rate=1000, default_burst=1)
        limiter.acquire("https://a.example/1")
        assert limiter.acquire("https://b.example/1") == 0.0


class TestRetryPolicy:
    def test_backoff_grows_and_is_capped(self) -> None:
        policy = RetryPolicy(attempts=5, base_delay_s=1.0, max_delay_s=5.0, factor=2.0, jitter=0.0)
        assert [policy.delay_for(i) for i in range(4)] == [1.0, 2.0, 4.0, 5.0]

    def test_retry_after_wins(self) -> None:
        policy = RetryPolicy(max_delay_s=30.0)
        assert policy.delay_for(0, retry_after_s=12.0) == 12.0

    def test_retry_after_is_capped(self) -> None:
        policy = RetryPolicy(max_delay_s=10.0)
        assert policy.delay_for(0, retry_after_s=600.0) == 10.0

    def test_jitter_reduces_delay(self) -> None:
        policy = RetryPolicy(base_delay_s=10.0, jitter=0.5)
        assert 5.0 <= policy.delay_for(0) <= 10.0


class TestRetryLoop:
    """The retry loop must be exercised without any real socket."""

    def _transport(self) -> HttpxTransport:
        return HttpxTransport(retry=RetryPolicy(attempts=3, base_delay_s=0.0), sleep=lambda s: None)

    def test_retries_on_retryable_status_then_succeeds(self, monkeypatch) -> None:
        tx = self._transport()
        responses = iter(
            [
                TransportResponse(url="u", status=503, text="busy"),
                TransportResponse(url="u", status=200, text="ok"),
            ]
        )
        monkeypatch.setattr(tx, "_single", lambda *a, **k: next(responses))
        out = tx.request("GET", "https://example.com/x")
        assert out.text == "ok"
        assert out.attempts == 2

    def test_gives_up_after_the_last_attempt(self, monkeypatch) -> None:
        tx = self._transport()
        monkeypatch.setattr(
            tx, "_single", lambda *a, **k: TransportResponse(url="u", status=503, text="busy")
        )
        with pytest.raises(TransportError) as exc:
            tx.request("GET", "https://example.com/x")
        assert exc.value.status == 503

    def test_non_retryable_status_fails_fast(self, monkeypatch) -> None:
        tx = self._transport()
        calls = {"n": 0}

        def single(*args, **kwargs):
            calls["n"] += 1
            return TransportResponse(url="u", status=404, text="nope")

        monkeypatch.setattr(tx, "_single", single)
        with pytest.raises(TransportError):
            tx.request("GET", "https://example.com/x")
        assert calls["n"] == 1

    def test_transient_transport_error_is_retried(self, monkeypatch) -> None:
        tx = self._transport()
        calls = {"n": 0}

        def single(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TransportError("timeout", transient=True)
            return TransportResponse(url="u", status=200, text="ok")

        monkeypatch.setattr(tx, "_single", single)
        assert tx.request("GET", "https://example.com/x").text == "ok"
        assert calls["n"] == 2

    def test_non_transient_error_is_not_retried(self, monkeypatch) -> None:
        tx = self._transport()
        calls = {"n": 0}

        def single(*args, **kwargs):
            calls["n"] += 1
            raise TransportError("bad request", transient=False)

        monkeypatch.setattr(tx, "_single", single)
        with pytest.raises(TransportError):
            tx.request("GET", "https://example.com/x")
        assert calls["n"] == 1
