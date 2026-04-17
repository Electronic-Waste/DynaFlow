import os
import subprocess
import argparse

model_name_to_short_name = {
    "deepseek-ai/DeepSeek-V2-Lite": "deepseek_v2_lite",
}

strategy_name_to_config = {
    "dbo": '{"scheduler_path": "../scheduler/vllm/dbo.py:DBOScheduler",'
    '"use_inductor": true, "min_nano_split_tokens": 2048,'
    '"max_num_splits": 2}',
    'original_dbo': '',
}


def run_with_retry(command, result_json, log_path, env=None, max_attempts=5):
    for attempt in range(max_attempts):
        if os.path.exists(result_json):
            os.remove(result_json)
        with open(log_path, "w") as f:
            print(f"Running command (attempt {attempt + 1}): {' '.join(command)}")
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        print(f"Attempt {attempt + 1} failed (exit code {result.returncode})")
    raise RuntimeError(f"Command failed after {max_attempts} attempts: {' '.join(command)}")


def create_parser():
    parser = argparse.ArgumentParser(description="EP Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        choices=model_name_to_short_name.keys(),
        required=True,
        help="Model name for benchmarking",
    )
    parser.add_argument("--dp-size", type=int, required=True, help="Data parallel size")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=strategy_name_to_config.keys(),
        required=False,
        help="Strategy for benchmarking",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fixed", "dataset"],
        required=True,
        help="Benchmark mode",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    model_name = str(args.model)
    model_short_name = model_name_to_short_name[model_name]
    dp_size = int(args.dp_size)
    mode = str(args.mode)
    strategy = str(args.strategy) if args.strategy else "none"
    testsuite_name = (
        f"vllm_ep_{strategy}/{model_short_name}"
        if strategy != "none"
        else f"vllm_ep/{model_short_name}"
    )
    dirname = f"../results/{testsuite_name}"
    os.makedirs(f"{dirname}/log", exist_ok=True)

    env = os.environ.copy()
    env["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    env["VLLM_ATTENTION_BACKEND"] = "CUTLASS_MLA"

    if mode == "fixed":
        input_output_lengths = [
            (512, 128),
            (1024, 128),
            (2048, 128),
        ]

        for input_len, output_len in input_output_lengths:
            for i in range(10):
                testcase_name = (
                    f"{model_short_name}_dp_{dp_size}_"
                    f"input{input_len}_output{output_len}_iter{i}"
                )
                result_json = f"{dirname}/{testcase_name}.json"
                command = [
                    "python",
                    "../scripts/bench_vllm_dp.py",
                    "--model",
                    model_name,
                    "--dp-size",
                    str(dp_size),
                    "--num-prompts",
                    "1024",
                    "--input-len",
                    str(input_len),
                    "--output-len",
                    str(output_len),
                    "--gpu-memory-utilization",
                    "0.9",
                    "--max-num-seqs",
                    "4096",
                    "--compilation-config",
                    '{"cudagraph_mode": "NONE"}',
                    "--output-json",
                    result_json,
                    "--enable-expert-parallel",
                ]
                if strategy == "dbo":
                    command += ["--dynaflow-config", strategy_name_to_config[strategy]]
                elif strategy == "original_dbo":
                    command += ["--enable-dbo"]
                run_with_retry(
                    command=command,
                    result_json=result_json,
                    log_path=f"{dirname}/log/{testcase_name}.log",
                    env=env,
                    max_attempts=1, # There is no need to retry for DBO
                )


    elif mode == "dataset":
        basedir = os.path.expanduser("~/.cache/dynaflow/eval_datasets")
        dataset_paths = {
            "sharegpt":  os.path.join(basedir, "sharegpt.json"),
            "lmsys":     os.path.join(basedir, "lmsys.json"),
            "splitwise": os.path.join(basedir, "splitwise.json"),
        }
        for dataset_label, dataset_path in dataset_paths.items():
            for i in range(5):
                testcase_name = (
                    f"{model_short_name}_dp_{dp_size}_{dataset_label}_iter{i}"
                )
                result_json = f"{dirname}/{testcase_name}.json"
                command = [
                    "python",
                    "../scripts/bench_vllm_dp.py",
                    "--model", model_name,
                    "--dp-size", str(dp_size),
                    "--dataset-name", "sharegpt",
                    "--dataset-path", dataset_path,
                    "--num-prompts", "4096",
                    "--gpu-memory-utilization", "0.9",
                    "--max-num-seqs", "4096",
                    "--compilation-config", '{"cudagraph_mode": "NONE"}',
                    "--output-json", result_json,
                    "--enable-expert-parallel",
                ]
                if strategy == "dbo":
                    command += ["--dynaflow-config", strategy_name_to_config[strategy]]
                elif strategy == "original_dbo":
                    command += ["--enable-dbo"]
                run_with_retry(
                    command=command,
                    result_json=result_json,
                    log_path=f"{dirname}/log/{testcase_name}.log",
                    env=env,
                    max_attempts=1, # There is no need to retry for DBO
                )


if __name__ == "__main__":
    main()
