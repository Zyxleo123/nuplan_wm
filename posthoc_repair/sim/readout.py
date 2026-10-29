"""Pool the chunked runs of posthoc_repair/sim/run_chunks.sh (RUN_ROOT/CHALLENGE/ARM/chunk_*) into one table per
arm, paired against a base arm.

The score is nuPlan's per-scenario CLS score (the aggregator's ``score`` column); the split score is their mean
over scenarios, which is what the weighted-average aggregator's ``final_score`` row is for one run (checked per
chunk). Paired differences are over the scenarios both arms finished, with a bootstrap CI over scenarios.

    python -m posthoc_repair.sim.readout \
        --run_root /data/user_data/$USER/nuplan_wm/runs/val14/closed_loop_nonreactive_agents --base dp
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

METRICS = ["no_ego_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
           "ego_is_making_progress", "time_to_collision_within_bound", "speed_limit_compliance",
           "ego_progress_along_expert_route", "ego_is_comfortable"]


def load_arm(arm_dir: str) -> tuple[pd.DataFrame, list[dict]]:
    rows, checks = [], []
    for chunk in sorted(glob.glob(os.path.join(arm_dir, "chunk_*"))):
        files = glob.glob(os.path.join(chunk, "aggregator_metric", "*.parquet"))
        if not os.path.exists(os.path.join(chunk, "done")) or not files:
            continue
        d = pd.read_parquet(files[0])
        per = d[d["log_name"].notna()].copy()
        final = d[d["scenario"] == "final_score"]["score"]
        checks.append({"chunk": os.path.basename(chunk), "n": len(per), "mean": float(per["score"].mean()),
                       "final_score": float(final.iloc[0]) if len(final) else float("nan")})
        per["chunk"] = os.path.basename(chunk)
        rows.append(per)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), checks


def repair_stats(arm_dir: str) -> dict | None:
    files = glob.glob(os.path.join(arm_dir, "chunk_*", "repair_stats", "*.jsonl"))
    if not files:
        return None
    s = pd.DataFrame([json.loads(line) for f in files for line in open(f)])
    # c_* are None when no object is in range (inf); a missing clearance is no violation
    viol0 = (s["c_edge0"].fillna(1.0) < 0) | (s["c_agent0"].fillna(1.0) < 0)
    viol1 = (s["c_edge"].fillna(1.0) < 0) | (s["c_agent"].fillna(1.0) < 0)
    return {"simulations": len(files), "replans": len(s), "charged": float((s["loss0"] > 0).mean()),
            "moved": float(s["moved"].mean()), "nonfinite": int(s["nonfinite"].sum()),
            "shift_m_p50_when_moved": float(s.loc[s["moved"], "shift_m"].median()) if s["moved"].any() else 0.0,
            "shift_m_p95_when_moved": float(s.loc[s["moved"], "shift_m"].quantile(0.95)) if s["moved"].any() else 0.0,
            "pred_violation_before": float(viol0.mean()), "pred_violation_after": float(viol1.mean()),
            # of the replans whose plan was predicted to violate, the share with no predicted violation after
            "violation_cleared": float((~viol1[viol0]).mean()) if viol0.any() else float("nan"),
            "ms_p50": float(s["ms"].median()), "ms_p99": float(s["ms"].quantile(0.99))}


def paired(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0, n_boot: int = 10000) -> dict:
    cols = ["scenario", "score"] + [k for k in METRICS if k in a and k in b]
    m = a[cols + ["scenario_type"]].merge(b[cols], on="scenario", suffixes=("_a", "_b"))
    d = (m["score_a"] - m["score_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, (n_boot, len(d))).mean(1) if len(d) else np.zeros(1)
    changed = []
    for i in np.argsort(-np.abs(d), kind="stable"):
        if abs(d[i]) <= 1e-9:
            break
        r = m.iloc[i]
        moved = {k: [float(r[k + "_b"]), float(r[k + "_a"])] for k in cols[2:] if abs(r[k + "_a"] - r[k + "_b"]) > 1e-9}
        changed.append({"scenario": r["scenario"], "type": r["scenario_type"], "score": [float(r["score_b"]),
                        float(r["score_a"])], "metrics": moved})
    return {"n": len(d), "diff": float(d.mean()) if len(d) else float("nan"),
            "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()), "changed": changed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_root", required=True)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--base", default="dp")
    p.add_argument("--out", default="")
    p.add_argument("--list", type=int, default=20, help="changed scenarios listed per arm")
    args = p.parse_args()
    arms = args.arms or sorted(d for d in os.listdir(args.run_root) if os.path.isdir(os.path.join(args.run_root, d)))
    data, out = {}, {"run_root": args.run_root, "arms": {}}
    for arm in arms:
        df, checks = load_arm(os.path.join(args.run_root, arm))
        data[arm] = df
        bad = [c for c in checks if abs(c["mean"] - c["final_score"]) > 1e-6]
        rec = {"scenarios": len(df), "chunks": len(checks), "final_score_mismatch": bad}
        if len(df):
            rec["score"] = float(df["score"].mean())
            rec.update({k: float(df[k].mean()) for k in METRICS if k in df})
        rec["repair"] = repair_stats(os.path.join(args.run_root, arm))
        out["arms"][arm] = rec
    if args.base in data and len(data[args.base]):
        for arm in arms:
            if arm != args.base and len(data[arm]):
                out["arms"][arm]["vs_" + args.base] = paired(data[arm], data[args.base])

    lines = [f"# {args.run_root}", "", "| arm | n | score | " + " | ".join(m.replace("_", " ") for m in METRICS)
             + f" | vs {args.base} (CI) | better / worse |", "|" + "---|" * (len(METRICS) + 5)]
    for arm, rec in out["arms"].items():
        if not rec["scenarios"]:
            lines.append(f"| {arm} | 0 |" + " |" * (len(METRICS) + 3))
            continue
        vs = rec.get("vs_" + args.base)
        vs_s = f"{100 * vs['diff']:+.2f} [{100 * vs['ci95'][0]:+.2f}, {100 * vs['ci95'][1]:+.2f}]" if vs else ""
        bw = f"{vs['better']} / {vs['worse']}" if vs else ""
        lines.append(f"| {arm} | {rec['scenarios']} | {100 * rec['score']:.2f} | "
                     + " | ".join(f"{100 * rec.get(m, float('nan')):.2f}" for m in METRICS) + f" | {vs_s} | {bw} |")
    for arm, rec in out["arms"].items():
        if rec["repair"]:
            r = rec["repair"]
            lines.append(f"\n{arm} repair: {r['replans']} replans in {r['simulations']} simulations; charged "
                         f"{100 * r['charged']:.1f}%, moved {100 * r['moved']:.1f}%; predicted violation "
                         f"{100 * r['pred_violation_before']:.2f}% -> {100 * r['pred_violation_after']:.2f}% of replans "
                         f"({100 * r['violation_cleared']:.0f}% of violating plans cleared); shift p50 {r['shift_m_p50_when_moved']:.2f} m / p95 "
                         f"{r['shift_m_p95_when_moved']:.2f} m; {r['ms_p50']:.1f} ms p50, {r['ms_p99']:.0f} ms p99; "
                         f"nonfinite {r['nonfinite']}")
        vs = rec.get("vs_" + args.base)
        if vs and vs["changed"]:
            lines.append(f"\n{arm} vs {args.base}, scenarios whose score changed (largest first; metric: base -> arm):")
            for c in vs["changed"][:args.list]:
                ms = ", ".join(f"{k} {v[0]:.2f}->{v[1]:.2f}" for k, v in c["metrics"].items())
                lines.append(f"- {c['scenario']} ({c['type']}): {c['score'][0]:.3f} -> {c['score'][1]:.3f}; {ms}")
        if rec["final_score_mismatch"]:
            lines.append(f"\n{arm}: chunk final_score != mean of scenario scores: {rec['final_score_mismatch']}")
    text = "\n".join(lines)
    print(text)
    dest = args.out or os.path.join(args.run_root, "readout")
    json.dump(out, open(dest + ".json", "w"), indent=1)
    open(dest + ".md", "w").write(text + "\n")


if __name__ == "__main__":
    main()
