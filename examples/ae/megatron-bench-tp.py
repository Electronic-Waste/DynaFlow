import argparse
import os
import subprocess


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TORCHRUN = os.path.join(SCRIPT_DIR, "Megatron-LM", ".venv", "bin", "torchrun")
BENCH_SCRIPT = os.path.join(SCRIPT_DIR, "scripts", "bench_megatron.py")

model_name_to_short_name = {
    "gpt-small": "gpt_small",
    "llama-3-8b": "llama3_8b",
}


def run_with_retry(command, log_path, env=None, max_attempts=3):
    for attempt in range(max_attempts):
        with open(log_path, "w") as f:
            print(f"Running command (attempt {attempt + 1}): {' '.join(command)}")
            result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        print(f"Attempt {attempt + 1} failed (exit code {result.returncode})")
    raise RuntimeError(f"Command failed after {max_attempts} attempts: {' '.join(command)}")


def create_parser():
    parser = argparse.ArgumentParser(description="Megatron TP Benchmark")
    parser.add_argument("--model", type=str, choices=model_name_to_short_name.keys(), required=True,
                        help="Model configuration to benchmark.")
    parser.add_argument("--tp-size", type=int, required=True, help="Tensor parallel size (number of GPUs).")
    parser.add_argument("--task", type=str, choices=["inference", "training"], required=True)
    parser.add_argument("--strategy", type=str, choices=["none", "dynaflow"], default="none",
                        help="'none': plain Megatron baseline; 'dynaflow': DynaFlow overlap.")
    parser.add_argument("--nano-batches", type=int, default=2,
                        help="Number of nano-batches for DynaFlow (ignored when strategy=none).")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length.")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override number of transformer layers.")
    parser.add_argument("--batch-sizes", type=str, default="16,32,64,128",
                        help="Comma-separated list of batch sizes to test.")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps per run.")
    parser.add_argument("--trials", type=int, default=10,
                        help="Measurement trials per run.")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    model_name = args.model
    model_short_name = model_name_to_short_name[model_name]
    tp_size = args.tp_size
    task = args.task
    strategy = args.strategy
    nano_batches = args.nano_batches

    testsuite_name = (
        f"megatron_dynaflow_nb{nano_batches}/{model_short_name}"
        if strategy == "dynaflow"
        else f"megatron/{model_short_name}"
    )
    dirname = os.path.join(SCRIPT_DIR, "..", "results", testsuite_name)
    os.makedirs(os.path.join(dirname, "log"), exist_ok=True)

    env = os.environ.copy()

    mode = "dynaflow" if strategy == "dynaflow" else "megatron"
    nccl_channels = env.get("NCCL_MAX_NCHANNELS", "default")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    port_base = 29600

    for i_bs, bs in enumerate(batch_sizes):
        for i_iter in range(4):
            port = port_base + i_bs * 4 + i_iter
            testcase_name = (
                f"{model_short_name}_tp{tp_size}_ch{nccl_channels}_{task}_bs{bs}_iter{i_iter}"
            )
            log_path = os.path.join(dirname, "log", f"{testcase_name}.log")

            run_with_retry(
                command=[
                    TORCHRUN,
                    f"--nproc_per_node={tp_size}",
                    f"--master_port={port}",
                    BENCH_SCRIPT,
                    "--mode", mode,
                    "--task", task,
                    "--batch-size", str(bs),
                    "--nano-batches", str(nano_batches),
                    "--model", model_name,
                    "--seq-len", str(args.seq_len),
                    *(["--num-layers", str(args.num_layers)] if args.num_layers else []),
                    "--warmup-steps", str(args.warmup_steps),
                    "--trials", str(args.trials),
                ],
                log_path=log_path,
                env=env,
            )


if __name__ == "__main__":
    main()
