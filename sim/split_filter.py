"""Split a token-list scenario filter (e.g. val14.yaml) into chunk filters for resumable simulation.

Each chunk filter is the source filter with ``scenario_tokens`` cut down to the chunk's tokens and
``log_names`` set to just their logs, so building a chunk's scenarios opens a handful of logs instead of
every log of the split. Every log's tokens stay in one chunk, so the filter's per-log rules
(``timestamp_threshold_s``) select the same scenarios as one unchunked run.

    python -m sim.split_filter --filter diffusion_planner/config/scenario_filter/val14.yaml \
        --data_root /data/datasets/nuPlan-v1.1/nuplan-v1.1/splits/trainval --out DIR --chunks 16
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import yaml


def tokens_in_log(db_file: str, tokens: list[bytes]) -> list[str]:
    con = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        found = []
        for i in range(0, len(tokens), 500):
            part = tokens[i:i + 500]
            rows = con.execute(f"SELECT token FROM lidar_pc WHERE token IN ({','.join('?' * len(part))})", part)
            found += [bytes(r[0]).hex() for r in rows]
        return found
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--filter", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--splitter", default="", help="nuplan splitter yaml; restricts the logs searched to its "
                   "'val' split (the devkit's nuplan.yaml by default)")
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    src = yaml.safe_load(open(args.filter))
    tokens = [str(t) for t in src["scenario_tokens"]]
    dbs = sorted(glob.glob(os.path.join(args.data_root, "*.db")))
    split_file = args.splitter or os.path.join(os.path.dirname(__import__("nuplan").__file__),
                                               "planning/script/config/common/splitter/nuplan.yaml")
    log_names = src.get("log_names")
    if isinstance(log_names, str) and "log_splits.val" in log_names:
        val = set(yaml.safe_load(open(split_file))["log_splits"]["val"])
        dbs = [d for d in dbs if os.path.basename(d)[:-3] in val]
    elif isinstance(log_names, list):
        dbs = [d for d in dbs if os.path.basename(d)[:-3] in set(log_names)]
    wanted = [bytes.fromhex(t) for t in tokens]
    with ThreadPoolExecutor(args.threads) as ex:
        found = list(ex.map(lambda d: tokens_in_log(d, wanted), dbs))
    log_of = {t: os.path.basename(d)[:-3] for d, ts in zip(dbs, found) for t in ts}
    missing = [t for t in tokens if t not in log_of]
    print(f"{len(tokens)} tokens in {len(set(log_of.values()))} logs ({len(dbs)} searched), {len(missing)} missing")

    by_log = {}
    for t in tokens:
        if t in log_of:
            by_log.setdefault(log_of[t], []).append(t)
    # Greedy balance: biggest logs first, each into the currently smallest chunk.
    chunks = [[] for _ in range(min(args.chunks, len(by_log)))]
    for log in sorted(by_log, key=lambda k: (-len(by_log[k]), k)):
        min(chunks, key=lambda c: sum(len(by_log[g]) for g in c)).append(log)
    # Hydra finds `scenario_filter=chunk_000` in <search path>/scenario_filter/.
    os.makedirs(os.path.join(args.out, "scenario_filter"), exist_ok=True)
    for i, logs in enumerate(chunks):
        f = dict(src, scenario_tokens=[t for lg in sorted(logs) for t in by_log[lg]], log_names=sorted(logs))
        with open(os.path.join(args.out, "scenario_filter", f"chunk_{i:03d}.yaml"), "w") as fh:
            yaml.safe_dump(f, fh, sort_keys=False)
    json.dump({"filter": os.path.abspath(args.filter), "log_of": log_of, "missing": missing,
               "chunks": {f"chunk_{i:03d}": sorted(c) for i, c in enumerate(chunks)}},
              open(os.path.join(args.out, "tokens_logs.json"), "w"), indent=1)
    print(f"{len(chunks)} chunks, tokens per chunk {[sum(len(by_log[g]) for g in c) for c in chunks]}")


if __name__ == "__main__":
    main()
