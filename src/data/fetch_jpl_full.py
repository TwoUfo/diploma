from __future__ import annotations

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


API_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

EXISTING_FIELDS = [
    "spkid", "full_name", "pdes", "name", "prefix",
    "neo", "pha",
    "H", "diameter", "albedo", "diameter_sigma",
    "orbit_id", "epoch", "epoch_mjd", "epoch_cal", "equinox",
    "e", "a", "q", "i", "om", "w", "ma", "ad", "n",
    "tp", "tp_cal", "per", "per_y",
    "moid", "moid_ld",
    "sigma_e", "sigma_a", "sigma_q", "sigma_i", "sigma_om", "sigma_w",
    "sigma_ma", "sigma_ad", "sigma_n", "sigma_tp", "sigma_per",
    "class", "rms",
]

NEW_FIELDS = [
    "n_obs_used",     # ρ=-0.803 with H — the gold feature
    "data_arc",       # ρ=-0.644 with H
    "condition_code", # ρ=+0.307 with H
    "H_sigma",        # enables heteroscedastic loss on H (~50% coverage)
    "rot_per",        # rotation period (hours)
    "spec_B",         # SMASSII spectral type
    "spec_T",         # Tholen spectral type
    "G",              # phase-curve slope parameter
    "BV", "UB", "IR", # photometric colour indices
]

FIELDS = EXISTING_FIELDS + NEW_FIELDS

CLASSES = [
    "HYA", "IEO", "AST",
    "ATE", "TJN", "AMO", "TNO", "CEN",
    "APO", "MCA", "IMB", "OMB",
    "MBA",
]

OUTPUT = Path("data/raw/dataset.csv")
BACKUP = OUTPUT.with_suffix(".csv.bak")
TIMEOUT_PER_CLASS = 1800  # 30 minutes for MBA worst case
MAX_ATTEMPTS = 3
RETRY_DELAY = 10


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    icon = {"INFO": "  ", "STEP": "▸ ", "OK": "✓ ", "WARN": "⚠ ", "ERR": "✗ "}.get(level, "  ")
    print(f"[{ts}] {icon}{msg}", flush=True)


def fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def fetch_one_class(orbit_class: str) -> tuple[list[str], list[list]]:
    params = {
        "sb-kind": "a",
        "sb-class": orbit_class,
        "fields": ",".join(FIELDS),
    }
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prefix = f"class={orbit_class} (attempt {attempt}/{MAX_ATTEMPTS})"
        try:
            log(f"requesting {prefix}, timeout={TIMEOUT_PER_CLASS}s, fields={len(FIELDS)} …", "STEP")
            t0 = time.time()
            resp = requests.get(API_URL, params=params, timeout=TIMEOUT_PER_CLASS)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - t0
            n = data.get("count", 0)
            rows = data.get("data") or []
            rate = n / max(elapsed, 0.01)
            log(f"got {n:,} rows in {fmt_elapsed(elapsed)} ({rate:,.0f} rows/s)", "OK")
            return data["fields"], rows
        except requests.Timeout:
            log(f"TIMEOUT after {fmt_elapsed(TIMEOUT_PER_CLASS)} on {prefix}", "ERR")
        except requests.RequestException as exc:
            log(f"HTTP error on {prefix}: {exc}", "ERR")
        if attempt < MAX_ATTEMPTS:
            log(f"retrying in {RETRY_DELAY}s …", "WARN")
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f"class={orbit_class}: all {MAX_ATTEMPTS} attempts failed")


def main() -> int:
    log("=" * 68)
    log("JPL SBDB Query — full asteroid catalogue fetch")
    log(f"output       : {OUTPUT.resolve()}")
    log(f"fields ({len(FIELDS):>3}): {len(EXISTING_FIELDS)} existing + {len(NEW_FIELDS)} new")
    log(f"classes ({len(CLASSES):>2}): {', '.join(CLASSES)}")
    log("=" * 68)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if BACKUP.exists():
            BACKUP.unlink()
        OUTPUT.rename(BACKUP)
        log(f"existing dataset moved to {BACKUP.name}", "OK")

    grand_start = time.time()
    total_rows = 0
    fields_written = False
    failed: list[str] = []

    with OUTPUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        for i, cls in enumerate(CLASSES, 1):
            log("")
            log(f"━━━ [{i}/{len(CLASSES)}] class {cls} ━━━")
            try:
                fields, rows = fetch_one_class(cls)
            except RuntimeError as exc:
                log(str(exc), "ERR")
                failed.append(cls)
                continue

            if not fields_written:
                writer.writerow(fields)
                fields_written = True
            writer.writerows(rows)
            fh.flush()
            total_rows += len(rows)
            running = time.time() - grand_start
            log(f"cumulative : {total_rows:,} rows | elapsed {fmt_elapsed(running)}", "OK")

    log("")
    log("=" * 68)
    log(f"finished {len(CLASSES) - len(failed)}/{len(CLASSES)} classes "
        f"in {fmt_elapsed(time.time() - grand_start)}", "OK")
    log(f"rows written : {total_rows:,}")
    if OUTPUT.exists():
        log(f"file size    : {OUTPUT.stat().st_size / 1e6:.1f} MB")
    if failed:
        log(f"FAILED       : {failed}", "WARN")
        log("rerun to retry — backup remains at dataset.csv.bak", "WARN")
        return 1
    log("all classes fetched successfully", "OK")
    log("next: run `python -m src.data.preprocessing` to regenerate parquets", "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
