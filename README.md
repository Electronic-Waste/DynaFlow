# SchedFlow

SchedFlow is a programmable scheduling layer that decouples operator execution from model implementation, enabling flexible intra-device parallelism strategies. It allows you to implement techniques like batch splitting, compute-communication overlap, and operator multiplexing without modifying model code. The scheduling policy can be dynamically adjusted at runtime while maintaining compatibility with optimizations like CUDA graphs and TorchInductor.
