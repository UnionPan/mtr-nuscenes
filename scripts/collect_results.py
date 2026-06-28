#!/usr/bin/env python3
"""Aggregate per-run metrics into reports/results.md and reports/summary.json."""
import glob
import json
import os
import sys

RUNS = ["full", "no_mlm", "contrastive_only", "motion_only", "ctx_t4", "ctx_t8",
        "frozen_img", "adapt_img"]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fmt(x, p=3):
    try:
        return f"{float(x):.{p}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    summary = {}
    for r in RUNS:
        ev = load(os.path.join(root, "runs", r, "eval_metrics.json"))
        if ev:
            summary[r] = ev
    baseline = load(os.path.join(root, "runs", "baseline_frozen", "eval_metrics.json"))

    lines = ["# MTR — Results Summary\n"]

    # Retrieval / probe / motion table
    lines.append("## Retrieval, linear probe, and motion (clean val)\n")
    lines.append("| run | v2t R@1 | v2t R@5 | t2v R@1 | mean R@1 | ADE (m) | FDE (m) | "
                 "probe motion-state | probe pedestrian |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    if baseline:
        lp = baseline["linear_probe"]; mo = baseline["motion"]
        lines.append(f"| baseline (frozen mean-pool) | — | — | — | — | {fmt(mo['ade'],2)} | "
                     f"{fmt(mo['fde'],2)} | {fmt(lp['motion_state']['acc'])} | "
                     f"{fmt(lp['has_pedestrian']['acc'])} |")
    for r in RUNS:
        if r not in summary:
            continue
        m = summary[r]["main"]["val"]
        lp = summary[r]["main"]["linear_probe"]
        lines.append(
            f"| {r} | {fmt(m.get('v2t_R@1'))} | {fmt(m.get('v2t_R@5'))} | "
            f"{fmt(m.get('t2v_R@1'))} | {fmt(m.get('mean_R@1'))} | {fmt(m['ade'],2)} | "
            f"{fmt(m['fde'],2)} | {fmt(lp['motion_state']['acc'])} | "
            f"{fmt(lp['has_pedestrian']['acc'])} |")
    # majority baselines for probes
    if baseline:
        lp = baseline["linear_probe"]
        lines.append(f"\n_Probe majority-class baselines: motion-state "
                     f"{fmt(lp['motion_state']['majority'])}, pedestrian "
                     f"{fmt(lp['has_pedestrian']['majority'])}._\n")

    # Robustness (full model)
    if "full" in summary and "robustness" in summary["full"]:
        rob = summary["full"]["robustness"]
        lines.append("\n## Robustness (full model, val)\n")
        lines.append("| condition | mean R@1 | ADE (m) |")
        lines.append("|---|---|---|")
        for k, v in rob.items():
            lines.append(f"| {k} | {fmt(v.get('mean_R@1'))} | {fmt(v['ade'],2)} |")

    # Efficiency (full model)
    if "full" in summary and "efficiency" in summary["full"]:
        eff = summary["full"]["efficiency"]
        lines.append("\n## Efficiency (full model)\n")
        lines.append("| pipeline | throughput (clips/s) | latency (ms/clip) | peak mem (GB) |")
        lines.append("|---|---|---|---|")
        for k in ("full_image_to_temporal", "cached_feature_temporal"):
            e = eff[k]
            lines.append(f"| {k} | {fmt(e['throughput_clips_s'],0)} | "
                         f"{fmt(e['latency_ms_per_clip'],1)} | {fmt(e['peak_mem_gb'],2)} |")
        lines.append(f"\n_Efficiency config: {eff['config']}._")

    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    with open(os.path.join(root, "reports", "summary.json"), "w") as f:
        json.dump({"runs": summary, "baseline": baseline}, f, indent=2, default=float)
    with open(os.path.join(root, "reports", "results_tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
