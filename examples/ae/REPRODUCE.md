# DynaFlow vLLM TP Benchmark 复现指南

> 复现 `examples/ae/vllm-bench-tp.py`(vLLM baseline vs DynaFlow/NanoFlow)的通用指南,适用于不同 GPU 数量 / TP 配置 / 集群环境。文中所有命令以环境变量参数化,可直接交给 Claude Code 按步骤执行。
>
> 本文沉淀自一次真实复现(8×B200 共享服务器,单卡 TP=1),所有「坑」均为实际触发后定位修复的,见 §7 速查表。

## 0. 参数与前置条件

先确定三个参数(后文命令直接引用):

```bash
export DYNAFLOW_ROOT=/path/to/DynaFlow          # 本仓库路径
export MODEL=meta-llama/Meta-Llama-3-8B-Instruct # 测试模型,见下表
export TP_SIZE=2                                 # 张量并行度
export GPUS=0,1                                  # 用哪些卡(数量 = TP_SIZE)
```

论文/Makefile 的标准配置(`examples/ae/Makefile` 中 `VLLM_TP_RUNS`):

| 模型 | TP | 说明 |
|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | 2 | 最低硬件门槛,推荐起步 |
| `meta-llama/Meta-Llama-3-70B-Instruct` | 8 | |
| `Qwen/Qwen2.5-72B-Instruct` | 8 | |

注意事项:

- **TP=1 也能跑通,但 NanoFlow 不会有加速**(其收益来自 allreduce 与计算重叠,单卡无跨卡通信)。验证环境用 TP=1 没问题,复现加速比必须 TP≥2。
- **HF 模型访问**:Llama 系列是 gated 模型,需要 `huggingface-cli login` 或设置 `HF_TOKEN`;如集群有共享模型缓存,设置 `HF_HOME` 指向它。若缓存里只有同构变体(如 Llama-**3.1**-8B-Instruct),可在 `vllm-bench-tp.py` 的 `model_name_to_short_name` 字典中加一行该模型的条目来复用缓存(本次复现即如此做)。
- **共享集群**:开跑前先 `nvidia-smi` 确认目标卡空闲,并始终用 `CUDA_VISIBLE_DEVICES` 钉死卡,避免打扰他人任务。
- **磁盘**:vLLM 仓库 + venv + 预编译 wheel 约 20 GB;模型权重另计(8B ≈ 16 GB)。

## 1. 必须使用 `ae` 分支(Bug #1)

**Bug:** `main` 分支的 `examples/ae/patch/vllm.patch` 不完整——缺少新文件
`vllm/v1/worker/dynaflow.py`,打 patch 后引擎启动时报:

```
ModuleNotFoundError: No module named 'vllm.v1.worker.dynaflow'
```

此外 main 分支的 `vllm-bench-tp.py` 在不带 `--strategy` 时会向命令行塞空字符串参数(argparse 报 unrecognized arguments),且无失败重试。

**修复:** 使用 `ae` 分支(完整 patch + `run_with_retry` + 正确的参数拼接):

```bash
cd $DYNAFLOW_ROOT && git checkout ae
```

## 2. 搭建 vLLM 环境

```bash
cd $DYNAFLOW_ROOT/examples/ae

# 安装 uv(如无)
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

# 克隆 vLLM,钉死 commit,打 patch
# (patch 路径是 patch/vllm.patch;Makefile 里写的 ../change.patch 已过时)
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 2dda3e35d054235b0c2170df359b42ec25b4fe2c
git apply ../patch/vllm.patch
ls vllm/v1/worker/dynaflow.py        # 必须存在,否则说明用错了分支的 patch

# venv(Python 3.10)
cd .. && uv venv -p 3.10 && source .venv/bin/activate
```

### 2.1 安装预编译 vLLM(Bug #2,最大的坑)

**Bug:** Makefile 写的 `VLLM_PRECOMPILED_WHEEL_COMMIT=$(git rev-parse HEAD~1)` 在该
vLLM commit 的 setup.py 中**不被识别**(那是更新版本才加的变量)。setup.py 自动探测
base commit 失败后会**静默回退到 nightly wheel**——nightly 针对新版 torch 构建,
C++ ABI 不匹配,导致 `import vllm._C` 时进程直接 abort:

```
terminate called after throwing an instance of 'std::bad_alloc'
```

(特征:连 `vllm --help` 都崩;gdb 栈在 `TORCH_LIBRARY_init__C` → `registerKernel` →
`basic_string::_M_construct`。)

**修复:** 用 `VLLM_PRECOMPILED_WHEEL_LOCATION` 显式指定与源码 commit 匹配的 wheel
(此处用 HEAD~1 即 `d83f3f7c…` 的 wheel,与 Makefile 原意一致):

```bash
cd vllm
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/d83f3f7cb37a0f1861f16c84d529abcd54889885/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
uv pip install -e .
cd ..
```

> aarch64 集群请把 URL 末尾的 `manylinux1_x86_64` 换成 `manylinux2014_aarch64`。
> 预编译 wheel 面向 CUDA 12.8 / torch 2.8.0;非 NVIDIA 或更老 CUDA 的环境需走源码编译(慢但通用):`uv pip install -e . --no-build-isolation`。

**验证**(两个目录陷阱,见 Bug #5):

```bash
# 必须换到一个"干净"目录:不能在 examples/ae 下(./vllm 仓库目录会把已装的包
# 遮蔽成 namespace package),也别在 /tmp 这类可能有杂质 .py 文件的公共目录
mkdir -p ~/.cache/cleanwd && cd ~/.cache/cleanwd
python -c "import torch, vllm._C; print('_C OK')"   # 不能 bad_alloc
vllm --version    # 应输出 0.10.2rc3.dev456+g2dda3e35d…precompiled
```

### 2.2 降级 transformers(Bug #3)

**Bug:** 依赖解析默认装上 transformers 5.x,tokenizer 初始化报:

```
AttributeError: TokenizersBackend has no attribute all_special_tokens_extended
```

**修复:**

```bash
uv pip install "transformers==4.56.2"
```

### 2.3 安装 DynaFlow 与 flashinfer

scheduler(`examples/ae/scheduler/vllm/nanoflow.py`)依赖 `dynaflow` 包与 `flashinfer`:

```bash
cd $DYNAFLOW_ROOT
uv pip install -e . -p examples/ae/.venv/bin/python
uv pip install flashinfer-python -p examples/ae/.venv/bin/python
examples/ae/.venv/bin/python -c "import dynaflow, flashinfer; from flashinfer import green_ctx; print('OK')"
```

## 3. 冒烟测试(强烈建议,每条 ~1 分钟)

确认三件事:引擎能起、NanoFlow scheduler 能加载、能产出吞吐数字:

```bash
cd $DYNAFLOW_ROOT/examples/ae && source .venv/bin/activate
SMOKE="--model $MODEL --tensor-parallel-size $TP_SIZE \
  --num-prompts 8 --n 1 --input-len 128 --output-len 16 \
  --compilation-config {\"cudagraph_mode\":\"NONE\"}"

# baseline(无 DynaFlow)
CUDA_VISIBLE_DEVICES=$GPUS VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  vllm bench throughput $SMOKE

# NanoFlow
CUDA_VISIBLE_DEVICES=$GPUS VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  vllm bench throughput $SMOKE --dynaflow-config \
  "{\"scheduler_path\": \"$PWD/scheduler/vllm/nanoflow.py:NanoFlowScheduler\", \"use_inductor\": false, \"min_nano_split_tokens\": 4096, \"max_num_splits\": 2, \"use_ar_norm_fusion\": true}"
```

两条都应以 `Throughput: … requests/s, … total tokens/s` 结束。NanoFlow 那条日志里能看到 `Executing with SplitConfig: …` 即 scheduler 生效。

## 4. 正式实验

**必须从 `examples/ae/vllm/` 目录运行**(Makefile 约定;脚本中的 `../results` 才会落在 `examples/ae/results/`):

```bash
cd $DYNAFLOW_ROOT/examples/ae/vllm && source ../.venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPUS

# baseline(无 DynaFlow)
python ../vllm-bench-tp.py --model $MODEL --tp-size $TP_SIZE --mode fixed
# NanoFlow
python ../vllm-bench-tp.py --model $MODEL --tp-size $TP_SIZE --strategy nanoflow --mode fixed
```

- fixed 模式:3 种输入长度(512/1024/2048,output 128)× 4 iter = **每组 12 个 run**;每个 run 冷启动引擎并跑 1024 prompts(B200 单卡上约 1.5 分钟/run,一组约 20 分钟;其他硬件按比例估)
- 失败自动重试(最多 5 次);若 5 次仍失败,脚本抛 RuntimeError 终止——此时查对应 log
- 结果 JSON:`examples/ae/results/{vllm,vllm_nanoflow}/<model_short_name>/*.json`;逐 run 完整日志在同目录 `log/` 下
- 进度监控:`ls examples/ae/results/*/<model_short_name>/*.json | wc -l`
- 其他可选 strategy:`tokenweave`、`nanoflow_old`、`flux`(命令同上,换 `--strategy` 即可)
- `--mode dataset`(ShareGPT/LMSYS)需先用 `scripts/preprocess_datasets.py` 生成数据集到 `~/.cache/dynaflow/eval_datasets/`

## 5. 汇总对比

```bash
cd $DYNAFLOW_ROOT/examples/ae/results && python3 - <<'EOF'
import json, glob, re, statistics, os

SHORT = sorted({p.split('/')[1] for p in glob.glob('vllm/*/')})[0].rstrip('/') \
        if glob.glob('vllm/*/') else None
assert SHORT, '先跑完实验再汇总'

def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r'input(\d+)_output(\d+)_iter(\d+)', f)
        out.setdefault((int(m[1]), int(m[2])), []).append(json.load(open(f)))
    return out

base = load(f'vllm/{SHORT}/*.json')
nano = load(f'vllm_nanoflow/{SHORT}/*.json')
summary = {'model_short_name': SHORT, 'configs': []}
print(f"{'input/output':>13} | {'baseline tok/s':>15} | {'nanoflow tok/s':>15} | speedup")
for key in sorted(base):
    b = [r['tokens_per_second'] for r in base[key]]
    n = [r['tokens_per_second'] for r in nano[key]]
    bm, nm = statistics.mean(b), statistics.mean(n)
    print(f'{key[0]:>6}/{key[1]:<6} | {bm:>9.1f} ±{statistics.stdev(b):>4.0f} | {nm:>9.1f} ±{statistics.stdev(n):>4.0f} | {nm/bm:.3f}x')
    summary['configs'].append({'input_len': key[0], 'output_len': key[1],
        'baseline_tok_s': b, 'nanoflow_tok_s': n, 'speedup': nm/bm})
json.dump(summary, open('comparison_nanoflow_vs_baseline.json', 'w'), indent=2)
print('\n已写入 comparison_nanoflow_vs_baseline.json')
EOF
```

## 6. 参考数据点(校验量级用)

单张 B200、TP=1、Llama-3.1-8B-Instruct、fixed 模式下的实测(2026-06):
baseline 与 NanoFlow 均在 33k–36k tok/s,加速比 0.99–1.04×(TP=1 无通信可重叠,符合预期)。
若你的 TP≥2 结果中 NanoFlow 无加速,先确认日志里有 `SplitConfig` 且 `num_nano_batches > 1`(批量太小不会触发切分,`min_nano_split_tokens` 默认 4096)。

## 7. 坑清单(按症状速查)

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | `ModuleNotFoundError: vllm.v1.worker.dynaflow` | main 分支 patch 缺新文件 | 用 `ae` 分支的 patch(§1) |
| 2 | `std::bad_alloc`,连 `vllm --help` 都崩 | setup.py 不认 `VLLM_PRECOMPILED_WHEEL_COMMIT`,静默回退 nightly wheel,torch ABI 不匹配 | `VLLM_PRECOMPILED_WHEEL_LOCATION` 指定匹配 wheel 重装(§2.1) |
| 3 | `TokenizersBackend has no attribute all_special_tokens_extended` | transformers 5.x 太新 | 钉 `transformers==4.56.2`(§2.2) |
| 4 | baseline 报 unrecognized arguments | main 分支脚本向命令塞空字符串 | ae 分支已修复 |
| 5 | `vllm.__file__` 为 None / import 行为诡异 | 在 `examples/ae` 下运行时 `./vllm` 仓库目录遮蔽已安装包;公共目录(如 /tmp)可能有杂质 .py 遮蔽标准库 | 验证 import 时换干净目录(§2.1) |
| 6 | gated 模型 401 / 不想重复下载 | 无 token 或缓存里只有同构变体 | 配 `HF_TOKEN`/`HF_HOME`;或在脚本模型表加缓存变体条目(§0) |
| 7 | 启动即 OOM / 打扰他人任务 | 共享集群卡被占 | 先 `nvidia-smi` 查空闲卡,`CUDA_VISIBLE_DEVICES` 钉死(§0) |
| 8 | NanoFlow 与 baseline 持平(TP≥2) | 批量未达切分阈值,scheduler 未生效 | 查日志 `SplitConfig` 的 `num_nano_batches`;必要时调 `min_nano_split_tokens`(§6) |
