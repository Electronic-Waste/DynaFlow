def pack_tokens(num_tokens: int, cudagraph_capture_sizes: list[int]) -> int:
    """Pad token count up to the next capture size when using CUDA Graphs."""
    if num_tokens <= max(cudagraph_capture_sizes):
        return min(size for size in cudagraph_capture_sizes if size >= num_tokens)
    return num_tokens
