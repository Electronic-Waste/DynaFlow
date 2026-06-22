"""Aggregate all TP2 Llama-3.1-8B prefill results into one comparison table.

Pulls from results/<suite>/llama3.1_8b/ for each suite and computes mean tok/s
+ speedup vs vanilla vLLM baseline for the 3 prefill configs.
"""
import json
import glob
import statistics
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CONFIGS = [(512, 128), (1024, 128), (2048, 128)]
SUITES = [
    ("vllm", "baseline"),
    ("vllm_nanoflow", "nanoflow (overlap+fusion)"),
    ("vllm_nanoflow_nofusion", "nanoflow (overlap only)"),
    # NOTE: single-nano-batch falls back to engine.execute_single_batch, which
    # BYPASSES the DynaFlow scheduler -> the TokenWeave fused kernel never runs.
    # So this row is effectively vanilla vLLM, NOT TokenWeave's fused kernel.
    ("vllm_tokenweave_nooverlap", "tokenweave single-batch (== vanilla vLLM*)"),
    # Overlap path deadlocks: symm_mem.rendezvous() hangs on this box (no IMEX
    # channels for NVLS multicast). No data.
    ("vllm_tokenweave_overlap", "tokenweave (overlap+fusion) [DEADLOCK]"),
]


def load(suite, i, o):
    files = sorted(glob.glob(
        f"{BASE}/{suite}/llama3.1_8b/llama3.1_8b_tp_2_input{i}_output{o}_iter*.json"))
    vals = [json.load(open(f))["tokens_per_second"] for f in files]
    if not vals:
        return None
    return {"mean": statistics.mean(vals), "n": len(vals),
            "min": min(vals), "max": max(vals)}


def main():
    baseline = {c: load("vllm", *c) for c in CONFIGS}
    rows = []
    for suite, label in SUITES:
        row = {"suite": suite, "label": label, "by_config": {}}
        for c in CONFIGS:
            d = load(suite, *c)
            b = baseline[c]
            if d and b:
                row["by_config"][f"{c[0]}/{c[1]}"] = {
                    "tok_s": round(d["mean"], 1), "n": d["n"],
                    "speedup_vs_baseline": round(d["mean"] / b["mean"], 4),
                }
            else:
                row["by_config"][f"{c[0]}/{c[1]}"] = None
        rows.append(row)

    out = {"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "tp_size": 2,
           "gpus": "B200 x2 (GPU 4,5)", "rows": rows}
    out_path = f"{BASE}/comparison_tp2_all_strategies.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    col_keys = [f"{c[0]}/{c[1]}" for c in CONFIGS]
    hdr = f"{'strategy':<32}" + "".join(f"{k:<18}" for k in col_keys)
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        line = f"{row['label']:<32}"
        for k in col_keys:
            cell = row["by_config"][k]
            text = f"{cell['tok_s']} ({cell['speedup_vs_baseline']}x)" if cell else "N/A"
            line += f"{text:<18}"
        print(line)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
