# TokenWeave Kernels

This directory contains the TokenWeave kernels for vLLM.

## Usage

```bash
mkdir -p build
cd build
cmake .. -DVLLM_PYTHON_EXECUTABLE=$(which python)
make -j8
```

After the build is complete, please check if `_tokenweave_C.abi3.so` is generated in the `build` directory.
