"""TokenWeave fusion-only (no overlap) benchmark.

TokenWeave's fused collective kernel deadlocks under DynaFlow's 2-nano-batch
async overlap on this setup, so we force single-nano-batch by setting
min_nano_split_tokens very high. This measures the TokenWeave fused
allreduce+rmsnorm kernel WITHOUT overlap -- the analogue of
vllm_nanoflow_nofusion. Same 3 prefill configs / model / prompt counts.
"""
import os
import subprocess

AE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_SHORT_NAME = "llama3.1_8b"
TP_SIZE = 2

DYNAFLOW_CONFIG = (
    "{\"scheduler_path\":"
    f"\"{os.path.join(AE_DIR, 'scheduler', 'vllm', 'tokenweave.py')}:TokenWeaveScheduler\","
    "\"use_inductor\": false, \"min_nano_split_tokens\": 99999999,"
    "\"max_num_splits\": 2}"
)

INPUT_OUTPUT_LENGTHS = [
    (512, 128),
    (1024, 128),
    (2048, 128),
]


def run_with_retry(command, result_json, log_path, env=None, max_attempts=5):
    for attempt in range(max_attempts):
        if os.path.exists(result_json):
            os.remove(result_json)
        with open(log_path, "w") as f:
            print(f"Running command (attempt {attempt + 1}): {' '.join(command)}", flush=True)
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        print(f"Attempt {attempt + 1} failed (exit code {result.returncode})", flush=True)
    raise RuntimeError(f"Command failed after {max_attempts} attempts: {' '.join(command)}")


def main():
    dirname = os.path.join(AE_DIR, "results", f"vllm_tokenweave_nooverlap/{MODEL_SHORT_NAME}")
    os.makedirs(f"{dirname}/log", exist_ok=True)

    env = os.environ.copy()
    env["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    for input_len, output_len in INPUT_OUTPUT_LENGTHS:
        for i in range(4):
            testcase_name = f"{MODEL_SHORT_NAME}_tp_{TP_SIZE}_" \
                            f"input{input_len}_output{output_len}_iter{i}"
            result_json = f"{dirname}/{testcase_name}.json"
            run_with_retry(
                command=[
                    "vllm", "bench", "throughput",
                    "--model", MODEL_NAME,
                    "--tensor-parallel-size", str(TP_SIZE),
                    "--num-prompts", "1024",
                    "--n", "1",
                    "--input-len", str(input_len),
                    "--output-len", str(output_len),
                    "--compilation-config", "{\"cudagraph_mode\": \"NONE\"}",
                    "--dynaflow-config", DYNAFLOW_CONFIG,
                    "--output-json", result_json,
                ],
                result_json=result_json,
                log_path=f"{dirname}/log/{testcase_name}.log",
                env=env,
            )
            print(f"Done: {testcase_name}", flush=True)


if __name__ == "__main__":
    main()
