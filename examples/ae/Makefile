install:
	git clone https://github.com/vllm-project/vllm.git
	cd vllm
	git checkout 2dda3e35d054235b0c2170df359b42ec25b4fe2c
	git apply ../change.patch

build:
	uv venv -p 3.10
	source .venv/bin/activate
	cd vllm
	VLLM_PRECOMPILED_WHEEL_COMMIT=$(git rev-parse HEAD~1) VLLM_USE_PRECOMPILED=1 uv pip install -e .