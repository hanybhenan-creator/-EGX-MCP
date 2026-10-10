"""HTTP transport for the EGX data layer — injectable, bounded, polite.

Why not just call ``httpx.get`` inline?

* **Testability.** Parsers are the risky part of any scraper. With a
  ``Transport`` seam, every parser test replays a recorded response through
  :class:`FakeTransport` and runs with no network, no flakiness and no sleeps.
* **Bounded waiting.** Connection/read timeouts are explicit. The parent
  project already learned this lesson the hard way (see the socket-timeout note
  in ``screener_provider``): an upstream that accepts a connection and then
  goes quiet hangs a tool call forever.
* **Politeness.** ``egx.com.eg`` is a public service, not an API we are
  entitled to hammer. :class:`TokenBucket` enforces a per-host ceiling so a
  292-symbol refresh becomes a paced crawl instead of a burst.
* **Observability.** Every attempt returns a :class:`TransportResponse`
  carrying status, latency and byte count, which is what feeds the audit table
  and the provenance records.
"""

from __future__ import annotations

import json as _json
import random
import threading
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol
from urllib.parse import urlparse

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 EGX-MCP/1.0 (+data-sync)"
)

#: Statuses worth a retry: upstream throttling and transient gateway trouble.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})


class TransportError(Exception):
    """Any failure to obtain a response. Normalised across httpx/urllib."""

    def __init__(
        self,
        message: str,
        *,
        url: str = "",
        status: Optional[int] = None,
        transient: bool = True,
        retry_after_s: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status = status
        self.transient = transient
        self.retry_after_s = retry_after_s


@dataclass
class TransportResponse:
    """A completed attempt — success or not — with its observable cost."""

    url: str
    status: int
    text: str
    headers: dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    #: Requested URL before redirects/query normalisation.
    final_url: str = ""
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def bytes_received(self) -> int:
        return len(self.text.encode("utf-8", errors="replace"))

    def json(self) -> Any:
        return _json.loads(self.text)


class Transport(Protocol):
    """Minimal seam: one method, no session lifecycle to fake."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        timeout_s: Optional[float] = None,
    ) -> TransportResponse:  # pragma: no cover - protocol
        ...


# ── pacing ──────────────────────────────────────────────────────────────────


class TokenBucket:
    """Classic token bucket; thread-safe and monotonic-clock based."""

    def __init__(self, rate_per_s: float, burst: Optional[float] = None) -> None:
        if rate_per_s <= 0:
            raise ValueError("rate_per_s must be > 0")
        self.rate = float(rate_per_s)
        self.capacity = float(burst if burst is not None else max(1.0, rate_per_s))
        self._tokens = self.capacity
        self._last = _time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, *, timeout_s: float = 60.0) -> float:
        """Block until *tokens* are available. Returns seconds actually waited."""
        waited = 0.0
        deadline = _time.monotonic() + timeout_s
        while True:
            with self._lock:
                now = _time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate
            if _time.monotonic() + sleep_for > deadline:
                raise TransportError(
                    f"rate limit wait exceeded {timeout_s:.0f}s", transient=True
                )
            _time.sleep(sleep_for)
            waited += sleep_for


class HostRateLimiter:
    """Per-host token buckets, so one slow host cannot starve another."""

    def __init__(self, default_rate: float = 2.0, default_burst: float = 2.0) -> None:
        self._default_rate = default_rate
        self._default_burst = default_burst
        self._overrides: dict[str, tuple[float, float]] = {}
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def set_host_rate(self, host: str, rate_per_s: float, burst: Optional[float] = None) -> None:
        with self._lock:
            self._overrides[host.lower()] = (rate_per_s, burst or max(1.0, rate_per_s))
            self._buckets.pop(host.lower(), None)

    def _bucket(self, host: str) -> TokenBucket:
        host = host.lower()
        with self._lock:
            bucket = self._buckets.get(host)
            if bucket is None:
                rate, burst = self._overrides.get(
                    host, (self._default_rate, self._default_burst)
                )
                bucket = TokenBucket(rate, burst)
                self._buckets[host] = bucket
            return bucket

    def acquire(self, url: str, timeout_s: float = 60.0) -> float:
        host = urlparse(url).netloc or "unknown"
        return self._bucket(host).acquire(timeout_s=timeout_s)


# ── retry policy ────────────────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter and ``Retry-After`` support."""

    attempts: int = 3
    base_delay_s: float = 0.6
    max_delay_s: float = 20.0
    #: Multiplier applied to the delay per attempt.
    factor: float = 2.0
    #: Fraction of the computed delay to randomise away (0 = deterministic).
    jitter: float = 0.5

    def delay_for(self, attempt: int, retry_after_s: Optional[float] = None) -> float:
        if retry_after_s is not None:
            return max(0.0, min(retry_after_s, self.max_delay_s))
        raw = min(self.base_delay_s * (self.factor ** attempt), self.max_delay_s)
        if self.jitter:
            raw *= 1.0 - self.jitter * random.random()
        return raw


# ── real transport ──────────────────────────────────────────────────────────


class HttpxTransport:
    """Default network transport: bounded timeouts, pooling, pacing, backoff.

    Retries live here rather than in the sources so every source inherits the
    same behaviour and the retry cost shows up in one place in the audit log.
    """

    def __init__(
        self,
        *,
        user_agent: str = DEFAULT_UA,
        retry: Optional[RetryPolicy] = None,
        limiter: Optional[HostRateLimiter] = None,
        default_timeout_s: float = 15.0,
        proxy: Optional[str] = None,
        follow_redirects: bool = True,
        verify: bool = True,
        sleep: Callable[[float], None] = _time.sleep,
    ) -> None:
        self.retry = retry or RetryPolicy()
        self.limiter = limiter or HostRateLimiter()
        self.default_timeout_s = default_timeout_s
        self.user_agent = user_agent
        self.follow_redirects = follow_redirects
        self.verify = verify
        self.proxy = proxy
        self._sleep = sleep
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import httpx  # imported lazily: offline tests never need it

                    self._client = httpx.Client(
                        timeout=self.default_timeout_s,
                        follow_redirects=self.follow_redirects,
                        verify=self.verify,
                        proxy=self.proxy,
                        headers={
                            "User-Agent": self.user_agent,
                            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
                        },
                    )
        return self._client

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def __enter__(self) -> "HttpxTransport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── core ────────────────────────────────────────────────────────────────

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        timeout_s: Optional[float] = None,
    ) -> TransportResponse:
        last_error: Optional[TransportError] = None
        started = _time.monotonic()

        for attempt in range(self.retry.attempts):
            try:
                self.limiter.acquire(url)
                resp = self._single(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    data=data,
                    json_body=json_body,
                    timeout_s=timeout_s,
                )
            except TransportError as exc:
                last_error = exc
                if not exc.transient or attempt == self.retry.attempts - 1:
                    raise
                self._sleep(self.retry.delay_for(attempt, exc.retry_after_s))
                continue

            if resp.status in RETRYABLE_STATUS and attempt < self.retry.attempts - 1:
                retry_after = _parse_retry_after(resp.headers)
                self._sleep(self.retry.delay_for(attempt, retry_after))
                continue

            resp.latency_ms = (_time.monotonic() - started) * 1000.0
            resp.attempts = attempt + 1
            if not resp.ok:
                raise TransportError(
                    f"HTTP {resp.status} for {url}",
                    url=url,
                    status=resp.status,
                    transient=resp.status in RETRYABLE_STATUS,
                )
            return resp

        raise last_error or TransportError(f"request failed: {url}", url=url)

    def _single(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]],
        params: Optional[Mapping[str, Any]],
        data: Optional[Mapping[str, Any]],
        json_body: Any,
        timeout_s: Optional[float],
    ) -> TransportResponse:
        import httpx

        try:
            r = self._get_client().request(
                method.upper(),
                url,
                headers=dict(headers or {}),
                params=dict(params or {}),
                data=dict(data or {}) if data else None,
                json=json_body,
                timeout=timeout_s or self.default_timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise TransportError(f"timeout after {timeout_s or self.default_timeout_s}s: {url}", url=url) from exc
        except httpx.HTTPError as exc:  # connection reset, DNS, TLS, …
            raise TransportError(f"{type(exc).__name__}: {exc}", url=url) from exc

        return TransportResponse(
            url=url,
            final_url=str(r.url),
            status=r.status_code,
            text=r.text,
            headers={k.lower(): v for k, v in r.headers.items()},
        )


def _parse_retry_after(headers: Mapping[str, str]) -> Optional[float]:
    """``Retry-After`` in seconds form; HTTP-date form is ignored (rare here)."""
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return None


# ── test transport ──────────────────────────────────────────────────────────


@dataclass
class FakeCall:
    method: str
    url: str
    params: dict[str, Any]
    headers: dict[str, str]


class FakeTransport:
    """Deterministic transport for hermetic tests.

    Routes are matched by ``substring in url`` in insertion order, so the most
    specific route should be registered first. A route value may be:

    * ``TransportResponse`` — returned as-is;
    * ``str`` — a 200 text response;
    * ``Exception`` — raised (use :class:`TransportError` for retry paths);
    * ``Callable[[FakeCall], Any]`` — computed response.
    """

    def __init__(self) -> None:
        self._routes: list[tuple[str, Any]] = []
        self.calls: list[FakeCall] = []
        self.delays: list[float] = []

    # ── registration ────────────────────────────────────────────────────────

    def add(self, url_contains: str, response: Any) -> "FakeTransport":
        self._routes.append((url_contains, response))
        return self

    def add_text(self, url_contains: str, text: str, status: int = 200) -> "FakeTransport":
        return self.add(
            url_contains,
            TransportResponse(url=url_contains, status=status, text=text),
        )

    def add_json(self, url_contains: str, payload: Any, status: int = 200) -> "FakeTransport":
        return self.add_text(url_contains, _json.dumps(payload), status=status)

    def add_file(self, url_contains: str, path: Any, status: int = 200) -> "FakeTransport":
        from pathlib import Path as _Path

        return self.add_text(url_contains, _Path(path).read_text(encoding="utf-8"), status=status)

    # ── transport protocol ──────────────────────────────────────────────────

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        data: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
        timeout_s: Optional[float] = None,
    ) -> TransportResponse:
        call = FakeCall(
            method=method.upper(),
            url=url,
            params=dict(params or {}),
            headers=dict(headers or {}),
        )
        self.calls.append(call)

        for needle, response in self._routes:
            if needle not in url:
                continue
            if isinstance(response, Exception):
                raise response
            if isinstance(response, TransportResponse):
                if response.url in ("", needle):
                    response.url = url
                return response
            if callable(response):
                produced = response(call)
                if isinstance(produced, str):
                    return TransportResponse(url=url, status=200, text=produced)
                if isinstance(produced, TransportResponse):
                    return produced
                if isinstance(produced, Exception):
                    raise produced
                return TransportResponse(url=url, status=200, text=_json.dumps(produced))
            if isinstance(response, str):
                return TransportResponse(url=url, status=200, text=response)
            return TransportResponse(url=url, status=200, text=_json.dumps(response))

        raise TransportError(f"FakeTransport: no route matched {url}", url=url, transient=False)

    # ── assertions helpers ──────────────────────────────────────────────────

    def urls(self) -> list[str]:
        return [c.url for c in self.calls]

    def called_with(self, needle: str) -> bool:
        return any(needle in c.url for c in self.calls)

    def route_count(self, needle: str) -> int:
        return sum(1 for c in self.calls if needle in c.url)


def text_response(body: str, status: int = 200, url: str = "") -> TransportResponse:
    """Shorthand used throughout the tests."""
    return TransportResponse(url=url, status=status, text=body)


def json_response(payload: Any, status: int = 200, url: str = "") -> TransportResponse:
    return TransportResponse(url=url, status=status, text=_json.dumps(payload))
