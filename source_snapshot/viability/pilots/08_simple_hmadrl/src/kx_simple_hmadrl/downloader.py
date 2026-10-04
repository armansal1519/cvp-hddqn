from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import http.client
import json
import lzma
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
import time as time_module
from typing import Callable, Iterable
from urllib.error import HTTPError
from urllib.request import Request
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
import pandas as pd


BASE_URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
RECORD = struct.Struct(">3I2f")
PRICE_SCALE = 1_000.0
CHICAGO = ZoneInfo("America/Chicago")
UTC = timezone.utc
PIPELINE_CUTOFF = datetime(2025, 9, 1, tzinfo=UTC)
SCHEMA = "kx_dukascopy_xauusd_monthly_v1"


class DownloadError(RuntimeError):
    pass


class PersistentHTTPSOpener:
    """Thread-local HTTP/1.1 connections avoid one TLS handshake per hour."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _connection(self, host: str, timeout: float) -> http.client.HTTPSConnection:
        connection = getattr(self._local, "connection", None)
        connection_host = getattr(self._local, "host", None)
        if connection is None or connection_host != host:
            connection = http.client.HTTPSConnection(host, timeout=timeout)
            self._local.connection = connection; self._local.host = host
        return connection

    def _reset(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
        self._local.connection = None; self._local.host = None

    def __call__(self, request: Request, *, timeout: float = 60) -> http.client.HTTPResponse:
        parsed = urlsplit(request.full_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("persistent downloader accepts HTTPS URLs only")
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        connection = self._connection(parsed.hostname, timeout)
        try:
            connection.request("GET", path, headers=dict(request.header_items()))
            response = connection.getresponse()
        except Exception:
            self._reset(); raise
        if response.status >= 400:
            payload = response.read(); headers = response.headers; reason = response.reason
            if response.status >= 500:
                self._reset()
            raise HTTPError(request.full_url, response.status, str(reason), headers, None)
        return response


class HTTP2Opener:
    """Thread-safe persistent HTTP/2 transport for Dukascopy's public feed."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            http2=True,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
        )

    def __call__(self, request: Request, *, timeout: float = 60) -> httpx.Response:
        response = self.client.get(
            request.full_url, headers=dict(request.header_items()), timeout=timeout,
        )
        if response.status_code >= 400:
            status = int(response.status_code); headers = response.headers
            reason = response.reason_phrase; response.close()
            raise HTTPError(request.full_url, status, reason, headers, None)
        return response

    def close(self) -> None:
        self.client.close()


@dataclass(frozen=True, order=True)
class Hour:
    instant: datetime

    def __post_init__(self) -> None:
        if self.instant.tzinfo is None or self.instant.minute or self.instant.second or self.instant.microsecond:
            raise ValueError("hour must be timezone-aware and hour aligned")

    @property
    def relative_path(self) -> Path:
        value = self.instant.astimezone(UTC)
        return Path(f"{value.year:04d}/{value.month - 1:02d}/{value.day:02d}/{value.hour:02d}h_ticks.bi5")

    @property
    def month(self) -> str:
        return self.instant.strftime("%Y-%m")


def iter_hours(start: datetime, end: datetime) -> Iterable[Hour]:
    cursor = start.astimezone(UTC)
    terminal = end.astimezone(UTC)
    while cursor < terminal:
        yield Hour(cursor)
        cursor += timedelta(hours=1)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(destination: Path, value: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(value)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_manifest(root: str | Path, manifest: dict[str, object]) -> bool:
    directory = Path(root)
    try:
        months = manifest["months"]
        if not isinstance(months, list):
            return False
        for month in months:
            if not isinstance(month, dict):
                return False
            for path_key, hash_key in (("one_minute", "one_minute_sha256"), ("five_minute", "five_minute_sha256")):
                path = directory / str(month[path_key])
                if not path.is_file() or sha256_file(path) != month[hash_key]:
                    return False
            for source in month["sources"]:
                path = directory / str(source["path"])
                if not path.is_file() or sha256_file(path) != source["sha256"]:
                    return False
        return True
    except (KeyError, TypeError, OSError):
        return False


def validate_manifest_digest(root: str | Path) -> bool:
    directory = Path(root)
    manifest = directory / "manifest.json"
    digest = directory / "manifest.sha256"
    if not manifest.is_file() or not digest.is_file():
        return False
    return digest.read_text() == f"{sha256_file(manifest)}  {manifest.name}\n"


def decode_bi5(payload: bytes, hour: Hour) -> pd.DataFrame:
    try:
        raw = lzma.decompress(payload)
    except lzma.LZMAError as exc:
        raise DownloadError(f"malformed BI5 for {hour.relative_path}: decompression failed") from exc
    if not raw or len(raw) % RECORD.size:
        raise DownloadError(f"malformed BI5 for {hour.relative_path}: invalid record length")
    rows = list(RECORD.iter_unpack(raw))
    milliseconds = [row[0] for row in rows]
    if milliseconds != sorted(milliseconds) or milliseconds[0] < 0 or milliseconds[-1] >= 3_600_000:
        raise DownloadError(f"malformed BI5 for {hour.relative_path}: unordered timestamp offsets")
    ask = [row[1] / PRICE_SCALE for row in rows]
    bid = [row[2] / PRICE_SCALE for row in rows]
    if any(b > a or not math.isfinite(a + b) for a, b in zip(ask, bid, strict=True)):
        raise DownloadError(f"malformed BI5 for {hour.relative_path}: crossed or nonfinite quote")
    origin_ns = int(hour.instant.timestamp() * 1_000_000_000)
    return pd.DataFrame({
        "event_time": pd.to_datetime([origin_ns + ms * 1_000_000 for ms in milliseconds], utc=True),
        "bid": bid,
        "ask": ask,
        "ask_quote_volume": [float(row[3]) for row in rows],
        "bid_quote_volume": [float(row[4]) for row in rows],
    })


def aggregate_hour(ticks: pd.DataFrame) -> pd.DataFrame:
    source = ticks.copy()
    source["midpoint"] = (source["bid"] + source["ask"]) / 2.0
    indexed = source.set_index("event_time")
    result = indexed["midpoint"].resample("1min").ohlc()
    result["quote_count"] = indexed["midpoint"].resample("1min").count()
    result["bid_quote_volume"] = indexed["bid_quote_volume"].resample("1min").sum()
    result["ask_quote_volume"] = indexed["ask_quote_volume"].resample("1min").sum()
    return result.loc[result["quote_count"] > 0].reset_index()


def aggregate_five_minute(one_minute: pd.DataFrame) -> pd.DataFrame:
    indexed = one_minute.set_index("event_time").sort_index()
    result = indexed.resample("5min", label="right", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "quote_count": "sum", "bid_quote_volume": "sum", "ask_quote_volume": "sum",
    })
    return result.dropna(subset=["open", "high", "low", "close"]).reset_index()


def valid_session_rows(five_minute: pd.DataFrame, minimum_fraction: float = 0.95) -> tuple[pd.DataFrame, dict[str, int]]:
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum fraction must be in (0, 1]")
    frame = five_minute.copy()
    local = frame["event_time"].dt.tz_convert(CHICAGO)
    frame["session_date"] = local.dt.date
    local_clock = local.dt.time
    frame = frame.loc[
        (local.dt.weekday < 5)
        # A bar stamped 07:20 contains the 07:15-07:20 interval and is not
        # decision-eligible inside the registered session.
        & (local_clock > time(7, 20))
        & (local_clock <= time(12, 25))
    ].copy()
    expected = 61
    minimum = math.ceil(expected * minimum_fraction)
    counts = frame.groupby("session_date").size()
    accepted = counts.loc[counts >= minimum].index
    rejected = counts.loc[counts < minimum].index
    return frame.loc[frame["session_date"].isin(accepted)].reset_index(drop=True), {
        "accepted_sessions": int(len(accepted)), "rejected_sessions": int(len(rejected)),
        "minimum_rows": minimum, "expected_rows": expected,
    }


class DukascopyDownloader:
    def __init__(
        self, root: str | Path, *, workers: int = 4, retries: int = 4,
        opener: Callable[..., object] | None = None,
        sleeper: Callable[[float], None] = time_module.sleep,
    ) -> None:
        if workers < 1 or retries < 1:
            raise ValueError("workers and retries must be positive")
        self.root = Path(root).expanduser().resolve()
        self.workers = workers
        self.retries = retries
        self.opener = opener or HTTP2Opener()
        self.sleeper = sleeper
        self.bi5 = self.root / "bi5"
        self.monthly = self.root / "monthly"
        self.checkpoints = self.root / "checkpoints"
        self._request_lock = threading.Lock()
        self._next_request_time = 0.0
        self.minimum_request_interval_seconds = 0.5

    def _wait_for_request_slot(self) -> None:
        # Four requests may remain in flight, but starts are paced globally so
        # the public endpoint is not hit with an unbounded burst.
        with self._request_lock:
            now = time_module.monotonic()
            delay = max(0.0, self._next_request_time - now)
            self._next_request_time = max(now, self._next_request_time) \
                + self.minimum_request_interval_seconds
        if delay:
            self.sleeper(delay)

    def _download_hour(self, hour: Hour) -> Path | None:
        destination = self.bi5 / hour.relative_path
        if destination.is_file():
            try:
                decode_bi5(destination.read_bytes(), hour)
                return destination
            except (DownloadError, OSError):
                # Preserve the bad transport artifact for diagnosis, but do
                # not let one interrupted/corrupt cache entry permanently
                # poison every resumable attempt.
                quarantine = destination.with_name(
                    f".{destination.name}.invalid.{time_module.time_ns()}"
                )
                os.replace(destination, quarantine)
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = Request(f"{BASE_URL}/{hour.relative_path.as_posix()}", headers={"User-Agent": "kx-simple-hmadrl/0.1"})
        for attempt in range(self.retries):
            retry_delay = float(2 ** attempt)
            try:
                self._wait_for_request_slot()
                response = self.opener(request, timeout=60)
                try:
                    payload = response.read()
                finally:
                    close = getattr(response, "close", None)
                    if close is not None:
                        close()
                # Dukascopy represents a legitimately absent weekend/holiday
                # hour as HTTP 200 with Content-Length: 0 rather than 404.
                # It is not a BI5 file and must not be retried as corruption.
                if not payload:
                    return None
                decode_bi5(payload, hour)
                partial = destination.with_suffix(destination.suffix + ".part")
                partial.write_bytes(payload)
                partial.replace(destination)
                return destination
            except HTTPError as exc:
                if exc.code == 404:
                    return None
                if exc.code in (429, 503):
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    retry_delay = max(
                        15.0 * (2 ** attempt),
                        float(retry_after) if retry_after and retry_after.isdigit() else 0.0,
                    )
                failure: Exception = exc
            except Exception as exc:  # transport and validation are both retryable
                failure = exc
            if attempt + 1 < self.retries:
                self.sleeper(retry_delay)
        raise DownloadError(f"failed after {self.retries} attempts: {hour.relative_path}") from failure

    def preflight(self, start: datetime) -> None:
        # The registered default begins on New Year's Day 2012 (a Sunday).
        # Locate the first actual quote before exercising a full 24-hour window.
        anchor: datetime | None = None
        # One liquid-hour probe per calendar day avoids issuing dozens of
        # slow zero-byte weekend requests merely to discover an anchor.
        terminal = min(start + timedelta(days=7), PIPELINE_CUTOFF)
        candidates = [Hour(start + timedelta(days=day, hours=12))
                      for day in range(7) if start + timedelta(days=day, hours=12) < terminal]
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            paths = list(pool.map(self._download_hour, candidates))
            usable = [candidate.instant for candidate, path in zip(candidates, paths, strict=True)
                      if path is not None]
            if usable:
                anchor = min(usable)
        if anchor is None:
            raise DownloadError("preflight found no quote file in the first seven days")
        hours = list(iter_hours(anchor, anchor + timedelta(hours=24)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            paths = list(pool.map(self._download_hour, hours))
        usable = [(hour, path) for hour, path in zip(hours, paths, strict=True) if path is not None]
        if not usable:
            raise DownloadError("24-hour preflight returned no valid BI5 files")
        for hour, path in usable:
            decode_bi5(path.read_bytes(), hour)

    def _process_month(self, hours: list[Hour]) -> dict[str, object]:
        month = hours[0].month
        checkpoint = self.checkpoints / f"{month}.json"
        one_path = self.monthly / "1min" / f"XAUUSD_{month}.parquet"
        five_path = self.monthly / "5min" / f"XAUUSD_{month}.parquet"
        if checkpoint.is_file() and one_path.is_file() and five_path.is_file():
            try:
                saved = json.loads(checkpoint.read_text())
            except (OSError, json.JSONDecodeError):
                saved = {}
            if (saved.get("schema") == SCHEMA and saved.get("month") == month
                    and validate_manifest(self.root, {"months": [saved]})):
                return saved
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            paths = list(pool.map(self._download_hour, hours))
        bars: list[pd.DataFrame] = []
        sources: list[dict[str, object]] = []
        for hour, path in zip(hours, paths, strict=True):
            if path is None:
                continue
            bars.append(aggregate_hour(decode_bi5(path.read_bytes(), hour)))
            sources.append({"path": str(path.relative_to(self.root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        if not bars:
            raise DownloadError(f"month {month} contains no usable quotes")
        one = pd.concat(bars, ignore_index=True).sort_values("event_time").drop_duplicates("event_time")
        five = aggregate_five_minute(one)
        accepted, coverage = valid_session_rows(five)
        if not coverage["accepted_sessions"]:
            raise DownloadError(f"month {month} contains no session meeting 95% coverage")
        # Reject a failed session from both products.  Keep all intraday bars
        # for accepted Chicago dates so slower BasePaper states have context.
        accepted_dates = set(accepted["session_date"])
        one_dates = one["event_time"].dt.tz_convert(CHICAGO).dt.date
        accepted_one = one.loc[one_dates.isin(accepted_dates)].reset_index(drop=True)
        _atomic_parquet(accepted_one, one_path)
        _atomic_parquet(accepted, five_path)
        product = {
            "schema": SCHEMA, "month": month, "sources": sources,
            "one_minute": str(one_path.relative_to(self.root)), "one_minute_sha256": sha256_file(one_path),
            "five_minute": str(five_path.relative_to(self.root)), "five_minute_sha256": sha256_file(five_path),
            "one_minute_rows": len(accepted_one), "five_minute_rows": len(accepted),
            "incomplete_sessions_removed_from_both_products": True, **coverage,
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = checkpoint.with_suffix(".json.part")
        partial.write_text(json.dumps(product, sort_keys=True, indent=2) + "\n")
        partial.replace(checkpoint)
        return product

    def _process_month_resilient(self, hours: list[Hour]) -> dict[str, object]:
        """Retry one checkpoint as a unit after endpoint-wide transient failures."""

        failure: Exception | None = None
        for attempt in range(4):
            try:
                return self._process_month(hours)
            except DownloadError as exc:
                failure = exc
                if attempt < 3:
                    self.sleeper(30.0 * (2 ** attempt))
        assert failure is not None
        raise DownloadError(
            f"month {hours[0].month} failed after four checkpoint attempts"
        ) from failure

    def run(self, start: datetime, end: datetime) -> Path:
        start, end = start.astimezone(UTC), end.astimezone(UTC)
        if start >= end or end > PIPELINE_CUTOFF:
            raise DownloadError("range must be nonempty and may not enter September-December 2025")
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            digest_path = self.root / "manifest.sha256"
            expected_digest = f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
            if (manifest.get("start") == start.date().isoformat()
                    and manifest.get("end") == end.date().isoformat()
                    and validate_manifest(self.root, manifest)):
                if not digest_path.is_file() or digest_path.read_text() != expected_digest:
                    _atomic_text(digest_path, expected_digest)
                return manifest_path
            raise DownloadError("immutable manifest has a different range or failed a declared hash")
        self.preflight(start)
        grouped: dict[str, list[Hour]] = {}
        for hour in iter_hours(start, end):
            grouped.setdefault(hour.month, []).append(hour)
        products = [self._process_month_resilient(hours) for hours in grouped.values()]
        pretraining_sessions = sum(int(p["accepted_sessions"]) for p in products if str(p["month"]) < "2025-01")
        if start <= datetime(2012, 1, 1, tzinfo=UTC) and end >= datetime(2025, 1, 1, tzinfo=UTC) and pretraining_sessions < 2500:
            raise DownloadError(f"pre-2025 support is insufficient: {pretraining_sessions} valid sessions")
        manifest = {
            "schema": SCHEMA, "instrument": "XAUUSD", "source_kind": "Dukascopy OTC tick-by-tick quotes",
            "start": start.date().isoformat(), "end": end.date().isoformat(),
            "locked_period_downloaded": False, "price_scale": PRICE_SCALE,
            "volume_semantics": {"quote_count": "proxy channel; not exchange-traded volume", "bid_quote_volume": "vendor quote volume", "ask_quote_volume": "vendor quote volume"},
            "pretraining_sessions": pretraining_sessions, "months": products,
        }
        if not validate_manifest(self.root, manifest):
            raise DownloadError("final source/product hash audit failed before manifest freeze")
        partial = manifest_path.with_suffix(".json.part")
        partial.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        partial.replace(manifest_path)
        _atomic_text(
            self.root / "manifest.sha256",
            f"{sha256_file(manifest_path)}  {manifest_path.name}\n",
        )
        return manifest_path


def _midnight(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time(), tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable Dukascopy XAUUSD BI5 acquisition and monthly bar conversion")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2025-09-01", help="exclusive; cannot exceed 2025-09-01")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root", default="viability/data/external/dukascopy_xauusd/direct_bi5_v1")
    args = parser.parse_args(argv)
    path = DukascopyDownloader(args.root, workers=args.workers).run(_midnight(args.start), _midnight(args.end))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
