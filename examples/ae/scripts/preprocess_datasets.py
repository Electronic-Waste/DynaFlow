"""
Download and preprocess evaluation datasets for DynaFlow AE benchmarks.

Datasets produced (all in ShareGPT JSON format):
  - sharegpt.json   : ShareGPT_V3 from HuggingFace
  - lmsys.json      : LMSYS-Chat-1M converted to ShareGPT format
  - splitwise.json  : Azure LLM Inference Traces with mock text matching the
                      real ContextTokens / GeneratedTokens distributions

Run once before benchmarking:
    python examples/ae/script/preprocess_datasets.py
"""

import csv
import json
import shutil
import urllib.request
import datasets
import huggingface_hub
from pathlib import Path

CACHE_DIR = Path("~/.cache/dynaflow/eval_datasets").expanduser()
SHAREGPT_PATH  = CACHE_DIR / "sharegpt.json"
LMSYS_PATH     = CACHE_DIR / "lmsys.json"
SPLITWISE_PATH = CACHE_DIR / "splitwise.json"

# Azure LLM Inference Trace URLs (Azure/AzurePublicDataset on GitHub)
_SPLITWISE_URLS = [
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv",
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv",
]

# Small fixed vocabulary — each word is typically 1 BPE token
_WORDS = [
    "the", "a", "in", "of", "to", "is", "it", "that", "and", "for",
    "on", "with", "as", "at", "be", "by", "this", "we", "or", "an",
]


def _tokens_to_text(n: int) -> str:
    """Return a string of approximately `n` tokens using a fixed vocabulary."""
    return " ".join(_WORDS[i % len(_WORDS)] for i in range(max(n, 1)))


def download_sharegpt() -> None:
    """Download ShareGPT_V3_unfiltered_cleaned_split.json from HuggingFace."""
    print("Downloading ShareGPT …")
    src = huggingface_hub.hf_hub_download(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
    )
    shutil.copy(src, SHAREGPT_PATH)
    print(f"  → {SHAREGPT_PATH}")


def download_lmsys() -> None:
    """Download LMSYS-Chat-1M and convert to ShareGPT JSON format."""
    print("Downloading LMSYS-Chat-1M …")
    ds = datasets.load_dataset("lmsys/lmsys-chat-1m", split="train")

    role_map = {"user": "human", "assistant": "gpt"}
    output = []
    for row in ds:
        turns = [
            {"from": role_map.get(t["role"], t["role"]), "value": t["content"]}
            for t in row["conversation"]
        ]
        if turns:
            output.append({"conversations": turns})

    with open(LMSYS_PATH, "w") as f:
        json.dump(output, f)
    print(f"  → {LMSYS_PATH} ({len(output)} conversations)")


def download_splitwise() -> None:
    """Download Azure LLM Inference Traces and build a ShareGPT-format JSON
    with mock text matching the real ContextTokens / GeneratedTokens lengths."""
    print("Downloading Splitwise (Azure LLM Inference) traces …")

    rows = []
    for url in _SPLITWISE_URLS:
        tmp = CACHE_DIR / ("_tmp_" + url.split("/")[-1])
        print(f"  Fetching {url.split('/')[-1]} …")
        urllib.request.urlretrieve(url, tmp)
        with open(tmp, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ctx = int(row["ContextTokens"])
                    gen = int(row["GeneratedTokens"])
                except (KeyError, ValueError):
                    continue
                if ctx > 0 and gen > 0:
                    rows.append((ctx, gen))
        tmp.unlink()

    output = [
        {
            "conversations": [
                {"from": "human", "value": _tokens_to_text(ctx)},
                {"from": "gpt",   "value": _tokens_to_text(gen)},
            ]
        }
        for ctx, gen in rows
    ]

    with open(SPLITWISE_PATH, "w") as f:
        json.dump(output, f)
    print(f"  → {SPLITWISE_PATH} ({len(output)} requests)")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (SHAREGPT_PATH,  download_sharegpt),
        (LMSYS_PATH,     download_lmsys),
        (SPLITWISE_PATH, download_splitwise),
    ]
    for path, fn in tasks:
        if path.exists():
            print(f"Skipping {path.name} (already exists)")
        else:
            fn()

    print("\nDone. Dataset files:")
    for path, _ in tasks:
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
