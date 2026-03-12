import os
import subprocess
import argparse

model_name_to_short_name = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_8b",
    "meta-llama/Meta-Llama-3-70B-Instruct": "llama3_70b",
    "Qwen/Qwen2.5-72B-Instruct": "qwen2.5_72b",
}

strategy_name_to_config = {
    "nanoflow": "{\"scheduler_path\":"
                f"\"{os.path.join(os.path.dirname(__file__), 'scheduler', 'vllm', 'nanoflow.py')}:NanoFlowScheduler\","
                "\"use_inductor\": false, \"min_nano_split_tokens\": 4096,"
                "\"max_num_splits\": 2,\"use_ar_norm_fusion\": true}",
    "tokenweave": "{\"scheduler_path\":"
                  f"\"{os.path.join(os.path.dirname(__file__), 'scheduler', 'vllm', 'tokenweave.py')}:TokenWeaveScheduler\","
                  "\"use_inductor\": false, \"min_nano_split_tokens\": 4096, \"max_num_splits\": 2}",
}

def create_parser():
    parser = argparse.ArgumentParser(description="TP Benchmark")
    parser.add_argument("--model", type=str, choices=model_name_to_short_name.keys(), required=True, help="Model name for benchmarking")
    parser.add_argument("--tp-size", type=int, required=True, help="Tensor parallel size")
    parser.add_argument("--strategy", type=str, choices=strategy_name_to_config.keys(), required=False, help="Strategy for benchmarking")
    parser.add_argument("--mode", type=str, choices=["fixed", "dataset"], required=True, help="Benchmark mode")
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    model_name = str(args.model)
    model_short_name = model_name_to_short_name[model_name]
    tp_size = int(args.tp_size)
    mode = str(args.mode)
    strategy = str(args.strategy) if args.strategy else "none"
    testsuite_name = f"vllm_{strategy}/{model_short_name}" if strategy != "none" else f"vllm/{model_short_name}"
    dirname = f"../results/{testsuite_name}"
    os.makedirs(f"{dirname}/log", exist_ok=True)

    env = os.environ.copy()
    env["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    if mode == "fixed":
        input_output_lengths = [
            (512, 128),
            (1024, 128),
            (2048, 128),
        ]

        for input_len, output_len in input_output_lengths:
            for i in range(10):
                testcase_name = f"{model_short_name}_tp_{tp_size}_" \
                                f"input{input_len}_output{output_len}_iter{i}"
                with open(f"{dirname}/log/{testcase_name}.log", "w") as f:
                    command = [
                            "vllm", "bench", "throughput",
                            "--model", model_name,
                            "--tensor-parallel-size", str(tp_size),
                            "--num-prompts", "1024",
                            "--n", "1",
                            "--input-len", str(input_len),
                            "--output-len", str(output_len),
                            "--compilation-config", f"{{\"cudagraph_mode\": \"NONE\"}}",
                            "--dynaflow-config" if strategy != "none" else "",
                            f"{strategy_name_to_config[strategy]}" if strategy != "none" else "",
                            f"--output-json", f"{dirname}/{testcase_name}.json",
                        ]
                    print(f"Running command: {' '.join(command)}")
                    subprocess.run(command, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
        




if __name__ == "__main__":
    main()
