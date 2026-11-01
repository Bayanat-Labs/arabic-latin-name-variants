"""Stream the Wikidata JSON dump once and keep the name strings of in-scope entities.

The dump (``wikidata-YYYYMMDD-all.json.gz``) is far larger than a laptop disk, so it is never
stored. One HTTP connection is read block by block, inflated in memory and split into lines. A
cheap byte test on each raw line (an Arabic-language label, alias or monolingual value) picks the
lines worth parsing; those, and lines with a ``P279`` (subclass of) claim, go in batches to a small
pool of parser processes. The parsers reduce each entity to a compact name record (see
``extract.py``) and each class to a row of the class table, which later puts items into class
groups.

The dump is a concatenation of gzip members. If the connection breaks, the reader reconnects with
an HTTP ``Range`` request from the last byte received (checked with ``If-Range`` against the ETag,
or against ``Last-Modified`` when the server sends no ETag), so the inflater keeps its state and
nothing is lost. If the whole process stops, it can resume from the start of the last saved gzip
member (``--resume``); checksums then cover only the part read in the final run, and the summary
says so. The reader is adapted from the Arab-World Wikidata slice builder.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import sys
import threading
import time
import zlib
from pathlib import Path

from .config import USER_AGENT

try:  # orjson is faster; the standard library is enough.
    import orjson

    def _loads(b: bytes):
        return orjson.loads(b)
except ImportError:  # pragma: no cover
    def _loads(b: bytes):
        return json.loads(b)

ITEM_PREFIX = b'{"type":"item","id":"Q'
PROPERTY_PREFIX = b'{"type":"property"'
P279_KEY = b'"P279":['
BLOCK = 4 << 20
CLASS_LANGS = ("en", "ar")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class StopRun(Exception):
    """The source refused the request (403/429) or changed; the run stops without retrying."""


# --------------------------------------------------------------------------- reading


class HttpSource(threading.Thread):
    """Read one URL on one connection, resuming with Range after a break."""

    def __init__(self, url: str, start: int, out: "queue.Queue", log, *, max_rate_mb: float | None,
                 expect_etag: str | None = None, hashes: bool = True, max_bytes: int | None = None,
                 test_break_every: int | None = None, max_attempts: int = 40, tee: str | None = None,
                 expect_size: int | None = None):
        super().__init__(daemon=True)
        self.url, self.offset, self.out, self.log = url, start, out, log
        self.max_rate = max_rate_mb * 1e6 if max_rate_mb else None
        self.etag = expect_etag
        self.expect_size = expect_size
        self.total: int | None = None
        self.last_modified: str | None = None
        self.hashers = {"md5": hashlib.md5(), "sha1": hashlib.sha1(), "sha256": hashlib.sha256()} if hashes and start == 0 else {}
        self.max_bytes = max_bytes
        self.test_break_every = test_break_every
        self.max_attempts = max_attempts
        self.retries = 0
        self.requests = 0
        self.error: str | None = None
        self.stop = threading.Event()
        self.tee = open(tee, "wb") if tee else None

    def run(self) -> None:
        import requests

        session = requests.Session()
        attempt = 0
        t_rate, n_rate = time.monotonic(), 0
        while not self.stop.is_set():
            headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
            if self.offset:
                headers["Range"] = f"bytes={self.offset}-"
                if self.etag:
                    headers["If-Range"] = self.etag
                elif self.last_modified:
                    headers["If-Range"] = self.last_modified
            try:
                self.requests += 1
                resp = session.get(self.url, headers=headers, stream=True, timeout=(30, 180))
            except requests.RequestException as exc:
                attempt = self._backoff(attempt, f"connect failed: {exc!r}")
                if attempt < 0:
                    return
                continue
            try:
                if resp.status_code in (403, 429):
                    raise StopRun(f"HTTP {resp.status_code} from the dump server; stopping")
                if resp.status_code >= 500:
                    resp.close()
                    attempt = self._backoff(attempt, f"HTTP {resp.status_code}")
                    if attempt < 0:
                        return
                    continue
                etag = resp.headers.get("ETag")
                if self.offset:
                    if resp.status_code != 206:
                        raise StopRun(f"expected 206 for a resumed read, got {resp.status_code} (file changed?)")
                    cr = resp.headers.get("Content-Range", "")
                    if not cr.startswith(f"bytes {self.offset}-"):
                        raise StopRun(f"unexpected Content-Range {cr!r} for offset {self.offset}")
                    self.total = int(cr.rsplit("/", 1)[1])
                elif resp.status_code != 200:
                    raise StopRun(f"HTTP {resp.status_code}")
                else:
                    self.total = int(resp.headers["Content-Length"])
                if self.expect_size is not None and self.total != self.expect_size:
                    raise StopRun(f"size {self.total} differs from the expected {self.expect_size}")
                lm = resp.headers.get("Last-Modified")
                if self.last_modified and lm and lm != self.last_modified:
                    raise StopRun(f"Last-Modified changed from {self.last_modified} to {lm}")
                if self.etag and etag and etag != self.etag:
                    raise StopRun(f"ETag changed from {self.etag} to {etag}")
                self.etag = self.etag or etag
                self.last_modified = self.last_modified or resp.headers.get("Last-Modified")
                broke_at = self.offset + self.test_break_every if self.test_break_every else None
                for blk in resp.iter_content(chunk_size=BLOCK):
                    if self.stop.is_set():
                        return
                    if not blk:
                        continue
                    for h in self.hashers.values():
                        h.update(blk)
                    if self.tee:
                        self.tee.write(blk)
                    self.out.put(("blk", self.offset, blk))
                    self.offset += len(blk)
                    attempt = 0
                    if self.max_rate:
                        n_rate += len(blk)
                        ahead = n_rate / self.max_rate - (time.monotonic() - t_rate)
                        if ahead > 0:
                            time.sleep(ahead)
                        if n_rate > 1e9:
                            t_rate, n_rate = time.monotonic(), 0
                    if self.max_bytes and self.offset >= self.max_bytes:
                        if self.tee:
                            self.tee.close()
                        self.out.put(("end", self.offset, b"max_bytes"))
                        return
                    if broke_at and self.offset >= broke_at:
                        raise ConnectionError("test break")
                if self.total is not None and self.offset >= self.total:
                    self.out.put(("end", self.offset, b""))
                    return
                attempt = self._backoff(attempt, f"connection closed at {self.offset} of {self.total}")
                if attempt < 0:
                    return
            except StopRun as exc:
                self.error = str(exc)
                self.log(f"STOP: {exc}")
                self.out.put(("error", self.offset, str(exc).encode()))
                return
            except Exception as exc:  # broken stream: resume with Range
                attempt = self._backoff(attempt, f"read failed at {self.offset}: {exc!r}")
                if attempt < 0:
                    return
            finally:
                resp.close()

    def _backoff(self, attempt: int, why: str) -> int:
        self.retries += 1
        if attempt >= self.max_attempts:
            self.error = f"giving up after {attempt} attempts: {why}"
            self.log(f"STOP: {self.error}")
            self.out.put(("error", self.offset, self.error.encode()))
            return -1
        wait = 1 if self.test_break_every else min(900, 30 * 2 ** min(attempt, 5))
        self.log(f"retry {attempt + 1} in {wait}s ({why})")
        time.sleep(wait)
        return attempt + 1


class FileSource(threading.Thread):
    """Read a local .json.gz file in blocks (tests and reruns from a saved copy)."""

    def __init__(self, path: str, start: int, out: "queue.Queue", log, **_):
        super().__init__(daemon=True)
        self.path, self.offset, self.out, self.log = path, start, out, log
        self.total = os.path.getsize(path)
        self.etag = None
        self.last_modified = None
        self.hashers = {"md5": hashlib.md5(), "sha1": hashlib.sha1(), "sha256": hashlib.sha256()} if start == 0 else {}
        self.retries = 0
        self.requests = 0
        self.error = None
        self.stop = threading.Event()

    def run(self) -> None:
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            while not self.stop.is_set():
                blk = fh.read(BLOCK)
                if not blk:
                    break
                for h in self.hashers.values():
                    h.update(blk)
                self.out.put(("blk", self.offset, blk))
                self.offset += len(blk)
        self.out.put(("end", self.offset, b""))


class Inflater(threading.Thread):
    """Inflate a multi-member gzip stream; report the compressed offset of each member boundary."""

    def __init__(self, inq: "queue.Queue", outq: "queue.Queue"):
        super().__init__(daemon=True)
        self.inq, self.outq = inq, outq
        self.members = 0
        self.error: str | None = None

    def run(self) -> None:
        d = zlib.decompressobj(31)
        fresh = True
        while True:
            kind, pos, blk = self.inq.get()
            if kind != "blk":
                if kind == "end" and not d.eof and not fresh and blk != b"max_bytes":
                    self.error = f"compressed stream ended inside a gzip member at {pos}"
                    self.outq.put(("error", pos, self.error.encode()))
                else:
                    self.outq.put((kind, pos, blk))
                return
            data = blk
            while data:
                try:
                    out = d.decompress(data)
                except zlib.error as exc:
                    self.error = f"inflate error near offset {pos}: {exc}"
                    self.outq.put(("error", pos, self.error.encode()))
                    return
                fresh = False
                if out:
                    self.outq.put(("data", pos, out))
                if d.eof:
                    rest = d.unused_data
                    used = len(data) - len(rest)
                    pos += used
                    self.members += 1
                    self.outq.put(("member", pos, b""))
                    d = zlib.decompressobj(31)
                    fresh = True
                    data = rest
                else:
                    data = b""


# --------------------------------------------------------------------------- writing


class ChunkWriter:
    """Write JSON Lines to numbered .jsonl.zst files, rotating after ``per_file`` lines."""

    def __init__(self, folder: Path, stem: str, seq: int, per_file: int):
        import zstandard

        self.folder, self.stem, self.seq, self.per_file = folder, stem, seq, per_file
        self._cctx = zstandard.ZstdCompressor(level=6)
        self._fh = None
        self._w = None
        self.n_in_file = 0
        folder.mkdir(parents=True, exist_ok=True)

    def path(self, seq: int) -> Path:
        return self.folder / f"{self.stem}_{seq:05d}.jsonl.zst"

    def write(self, line: bytes) -> None:
        if self._w is None:
            self._fh = open(self.path(self.seq).with_suffix(".zst.part"), "wb")
            self._w = self._cctx.stream_writer(self._fh, closefd=False)
        self._w.write(line)
        self._w.write(b"\n")
        self.n_in_file += 1
        if self.n_in_file >= self.per_file:
            self.close_file()

    def close_file(self) -> None:
        if self._w is None:
            return
        self._w.close()
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        part = self.path(self.seq).with_suffix(".zst.part")
        part.rename(self.path(self.seq))
        self._w = self._fh = None
        self.n_in_file = 0
        self.seq += 1


# --------------------------------------------------------------------------- workers

_KEEP: frozenset | None = None
_PERSON: frozenset = frozenset({"Q5"})
_EXCLUDE: frozenset = frozenset()


def _init_worker(keep: list[str] | None, person: list[str], exclude: list[str] | None = None) -> None:
    global _KEEP, _PERSON, _EXCLUDE
    _KEEP = frozenset(keep) if keep is not None else None
    _PERSON = frozenset(person) | {"Q5"}
    _EXCLUDE = frozenset(exclude or ())


def _work(batch: list[tuple[bytes, bool, bool]]) -> tuple[list[bytes], list[bytes], int]:
    """Parse a batch of raw lines; return (name records, class rows, parse errors)."""
    from .extract import extract

    recs, classes, errors = [], [], 0
    for line, cand, p279 in batch:
        try:
            ent = _loads(line)
        except Exception:
            errors += 1
            continue
        if p279:
            row = _class_row_from(ent)
            if row is not None:
                classes.append(row)
        if cand and ent.get("id") not in _EXCLUDE:
            rec = extract(ent, _KEEP, _PERSON)
            if rec is not None:
                recs.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":")).encode())
    return recs, classes, errors


def _class_row_from(ent: dict) -> bytes | None:
    parents = []
    for st in ent.get("claims", {}).get("P279", []):
        if st.get("rank") == "deprecated":
            continue
        ms = st.get("mainsnak", {})
        if ms.get("snaktype") != "value":
            continue
        v = ms.get("datavalue", {}).get("value")
        if isinstance(v, dict) and v.get("id"):
            parents.append(v["id"])
    if not parents:
        return None
    labels = ent.get("labels", {})
    row = {"id": ent["id"], "p279": parents}
    for lang in CLASS_LANGS:
        if lang in labels:
            row[lang] = labels[lang]["value"]
    return json.dumps(row, ensure_ascii=False).encode()


class InlineExecutor:
    """Runs submitted work at once in this process (``--workers 0``; tests and small files)."""

    def __init__(self, initializer=None, initargs=()):
        if initializer:
            initializer(*initargs)

    def submit(self, fn, *args):
        from concurrent.futures import Future

        fut: Future = Future()
        fut.set_result(fn(*args))
        return fut

    def shutdown(self, *a, **k):
        return None


def _read_list(path: str | None) -> list[str] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


# --------------------------------------------------------------------------- main loop


def run(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from .extract import arabic_candidate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.log", "a", buffering=1)

    def log(msg: str) -> None:
        logf.write(f"{_utc()} {msg}\n")

    state_path = out / "state.json"
    keep = _read_list(args.keep_classes)
    person = _read_list(args.person_classes) or []
    exclude = _read_list(args.exclude_qids) or []
    log(f"start: keep classes {len(keep) if keep is not None else 'all'}; person classes {len(person)}; "
        f"source {args.url or args.file}")

    st = {
        "resume_offset": 0, "carry_b64": "", "next_seq": 0, "lines": 0, "items": 0, "properties": 0,
        "other": 0, "candidates": 0, "kept": 0, "class_rows": 0, "parse_errors": 0, "members": 0,
        "decompressed_bytes": 0, "runs": [],
    }
    if args.resume and state_path.exists():
        st.update(json.loads(state_path.read_text()))
        for p in out.glob("*.jsonl.zst*"):
            if int(p.name.split("_")[1].split(".")[0]) >= st["next_seq"]:
                p.unlink()
        log(f"resume from compressed offset {st['resume_offset']:,} (seq {st['next_seq']})")
    elif (state_path.exists() or (out / "summary.json").exists()) and not args.resume:
        print("this folder holds an earlier run; pass --resume or use a new --out folder", file=sys.stderr)
        return 2

    q_raw: queue.Queue = queue.Queue(maxsize=args.queue_raw)
    q_txt: queue.Queue = queue.Queue(maxsize=args.queue_text)
    if args.file:
        src = FileSource(args.file, st["resume_offset"], q_raw, log)
    else:
        src = HttpSource(args.url, st["resume_offset"], q_raw, log, max_rate_mb=args.max_rate,
                         expect_etag=args.etag, max_bytes=args.max_bytes,
                         test_break_every=args.test_break_every, tee=args.test_tee,
                         expect_size=args.expect_size)
    inf = Inflater(q_raw, q_txt)
    rec_w = ChunkWriter(out, "names", st["next_seq"], args.per_file)
    class_w = ChunkWriter(out, "classes", st["next_seq"], 10**12)
    run_info = {"started_utc": _utc(), "start_offset": st["resume_offset"], "hashes_complete": st["resume_offset"] == 0}
    if args.workers > 0:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                   initargs=(keep, person, exclude))
    else:
        pool = InlineExecutor(initializer=_init_worker, initargs=(keep, person, exclude))
    pending: list = []
    batch: list = []
    batch_bytes = 0
    src.start()
    inf.start()

    carry = base64.b64decode(st["carry_b64"]) if st["carry_b64"] else b""
    lines = st["lines"]; items = st["items"]; props = st["properties"]; other = st["other"]
    cands = st["candidates"]; kept = st["kept"]; class_rows = st["class_rows"]; errors = st["parse_errors"]
    dec = st["decompressed_bytes"]
    last_qid = b""
    t0 = time.monotonic(); t_log = t0; off_log = st["resume_offset"]
    status = "ok"
    last_ckpt = st["resume_offset"]

    def collect(fut) -> None:
        nonlocal kept, class_rows, errors
        recs, classes, err = fut.result()
        for r in recs:
            rec_w.write(r)
        for c in classes:
            class_w.write(c)
        kept += len(recs); class_rows += len(classes); errors += err

    def flush_batch() -> None:
        nonlocal batch, batch_bytes
        if batch:
            pending.append(pool.submit(_work, batch))
            batch, batch_bytes = [], 0
        while len(pending) > args.max_pending:
            collect(pending.pop(0))

    def drain() -> None:
        flush_batch()
        while pending:
            collect(pending.pop(0))

    def save_state(boundary: int) -> None:
        drain()
        rec_w.close_file()
        class_w.close_file()
        seq = max(rec_w.seq, class_w.seq)
        rec_w.seq = class_w.seq = seq
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "resume_offset": boundary, "carry_b64": base64.b64encode(carry).decode(), "next_seq": seq,
            "lines": lines, "items": items, "properties": props, "other": other, "candidates": cands,
            "kept": kept, "class_rows": class_rows, "parse_errors": errors,
            "members": st["members"] + inf.members, "decompressed_bytes": dec, "runs": st["runs"],
            "saved_utc": _utc(),
        }, indent=1))
        tmp.replace(state_path)

    def progress() -> None:
        nonlocal t_log, off_log
        now = time.monotonic()
        rate = (src.offset - off_log) / max(now - t_log, 1e-6) / 1e6
        pct = 100.0 * src.offset / src.total if src.total else 0.0
        rec = {"utc": _utc(), "compressed_bytes": src.offset, "total_bytes": src.total, "pct": round(pct, 3),
               "net_mb_s": round(rate, 2), "decompressed_bytes": dec, "lines": lines, "items": items,
               "candidates": cands, "kept": kept, "class_rows": class_rows, "parse_errors": errors,
               "members": st["members"] + inf.members, "retries": src.retries, "requests": src.requests,
               "last_qid": last_qid.decode(), "elapsed_s": round(now - t0)}
        with open(out / "progress.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        eta = ""
        if src.total and rate > 0:
            eta = f"  eta {(src.total - src.offset) / (rate * 1e6) / 3600:.1f} h"
        log(f"{pct:6.2f}% {src.offset/1e9:8.2f} GB  {rate:6.1f} MB/s  items {items:,}  cand {cands:,}  "
            f"kept {kept:,}  classes {class_rows:,}  last {last_qid.decode()}  retries {src.retries}{eta}")
        t_log, off_log = now, src.offset

    try:
        while True:
            kind, pos, payload = q_txt.get()
            if kind == "data":
                dec += len(payload)
                data = carry + payload if carry else payload
                cut = data.rfind(b"\n")
                if cut < 0:
                    carry = data
                    continue
                carry = data[cut + 1:]
                for line in data[:cut].split(b"\n"):
                    lines += 1
                    if line.startswith(ITEM_PREFIX):
                        items += 1
                        cand = arabic_candidate(line)
                        p279 = args.class_table and P279_KEY in line
                        if cand or p279:
                            body = line[:-1] if line.endswith(b",") else line
                            batch.append((body, cand, p279))
                            batch_bytes += len(body)
                            if cand:
                                cands += 1
                                j = line.find(b'"', 22)
                                last_qid = line[21:j]
                            if batch_bytes >= args.batch_bytes:
                                flush_batch()
                    elif line.startswith(PROPERTY_PREFIX):
                        props += 1
                    elif line not in (b"[", b"]", b""):
                        other += 1
                if items == 0 and lines >= 10_000:
                    status = "error: no item lines in the first 10,000 lines (unexpected dump format)"
                    break
                if time.monotonic() - t_log >= args.log_every:
                    progress()
            elif kind == "member":
                if pos - last_ckpt >= args.checkpoint_every:
                    save_state(pos)
                    last_ckpt = pos
            elif kind == "end":
                if carry.strip() not in (b"", b"]"):
                    lines += 1
                    other += 1
                status = "ok" if payload != b"max_bytes" else "stopped_at_max_bytes"
                break
            elif kind == "error":
                status = "error: " + payload.decode()
                break
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        src.stop.set()
        try:
            drain()
        finally:
            pool.shutdown()
        rec_w.close_file()
        class_w.close_file()
        progress()

    run_info.update({"finished_utc": _utc(), "end_offset": src.offset, "status": status,
                     "retries": src.retries, "requests": src.requests})
    hashes = {k: h.hexdigest() for k, h in src.hashers.items()} if status == "ok" else {}
    summary = {
        "source_url": args.url or args.file, "etag": src.etag, "last_modified": src.last_modified,
        "total_bytes": src.total, "status": status, "hashes": hashes,
        "hashes_cover_whole_file": bool(hashes) and run_info["hashes_complete"],
        "lines": lines, "items": items, "properties": props, "other_lines": other,
        "candidate_lines": cands, "kept_records": kept, "class_rows": class_rows, "parse_errors": errors,
        "members": st["members"] + inf.members, "decompressed_bytes": dec,
        "keep_classes": len(keep) if keep is not None else None, "person_classes": len(person),
        "runs": st["runs"] + [run_info],
    }
    if args.expect_md5 and hashes:
        summary["md5_matches_published"] = hashes.get("md5") == args.expect_md5
    if args.expect_sha1 and hashes:
        summary["sha1_matches_published"] = hashes.get("sha1") == args.expect_sha1
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"ended with status {status}; kept {kept:,} records, {class_rows:,} class rows")
    return 0 if status in ("ok", "stopped_at_max_bytes") else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="dump URL (dated file, e.g. .../20261026/wikidata-20261026-all.json.gz)")
    src.add_argument("--file", help="local .json.gz file instead of a URL")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--keep-classes", help="text file of class QIDs to keep (P279 closure of the kept roots); "
                    "without it every item with a P31 value and Arabic and Latin names is kept")
    ap.add_argument("--person-classes", help="text file of human (Q5) and its subclasses")
    ap.add_argument("--etag", help="expected ETag (checked on each request)")
    ap.add_argument("--expect-size", type=int, help="expected size in bytes (checked on each request)")
    ap.add_argument("--expect-md5")
    ap.add_argument("--expect-sha1")
    ap.add_argument("--max-rate", type=float, default=40.0, help="MB/s cap on the download (default 40)")
    ap.add_argument("--workers", type=int, default=2, help="parser processes (default 2; 0 parses in this process)")
    ap.add_argument("--batch-bytes", type=int, default=8 << 20, help="raw bytes per parser batch")
    ap.add_argument("--max-pending", type=int, default=6, help="batches in flight before waiting")
    ap.add_argument("--per-file", type=int, default=200_000, help="records per output file")
    ap.add_argument("--log-every", type=float, default=180.0, help="seconds between progress lines")
    ap.add_argument("--checkpoint-every", type=float, default=2e9,
                    help="compressed bytes between resume points (taken at gzip member boundaries)")
    ap.add_argument("--exclude-qids", help="text file of QIDs whose name records are not written")
    ap.add_argument("--no-class-table", dest="class_table", action="store_false",
                    help="do not write the class table (when a class table of the same dump is at hand)")
    ap.add_argument("--queue-raw", type=int, default=32, help="compressed blocks held in memory (4 MiB each)")
    ap.add_argument("--queue-text", type=int, default=8, help="inflated blocks held in memory")
    ap.add_argument("--resume", action="store_true", help="resume from state.json in --out")
    ap.add_argument("--max-bytes", type=int, help="stop after this many compressed bytes (tests)")
    ap.add_argument("--test-break-every", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--test-tee", help=argparse.SUPPRESS)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
