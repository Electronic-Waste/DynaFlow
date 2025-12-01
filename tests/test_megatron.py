# Required modification to megatron-core 0.14.0:
# core/transformer/dot_product_attention.py
# Remove:
# with tensor_parallel.get_cuda_rng_tracker().fork():
# core/utils.py
# Remove:
# logging.warning

import os

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer import TransformerConfig

TRAIN_MODE = True


def setup_distributed() -> None:
    """Initialize distributed training."""
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
    
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    rng_tracker = get_cuda_rng_tracker()
    assert rng_tracker is not None
    rng_tracker.add('model-parallel-rng', 1234)


def create_model() -> GPTModel:
    """Create minimal GPT model using megatron-core."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=512,
        ffn_hidden_size=512,
        num_attention_heads=32,
        layernorm_epsilon=1e-5,
        add_bias_linear=False,
        add_qkv_bias=False,
    )
    
    layer_spec = get_gpt_layer_local_spec()
    
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=32768,
        max_sequence_length=4096,
        pre_process=True,
        post_process=True,
    ).cuda()
    
    from megatron.core.distributed import DistributedDataParallelConfig
    model.ddp_config = DistributedDataParallelConfig()
    
    return model


def create_optimizer(model: GPTModel) -> MegatronOptimizer:
    """Create optimizer using megatron-core."""
    optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=1e-3,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )
    
    optimizer = get_megatron_optimizer(
        config=optimizer_config,
        model_chunks=[model],
    )
    
    return optimizer


def create_dummy_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a dummy batch of data for training."""
    batch_size = 1
    seq_length = 128
    vocab_size = 32768
    
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long, device='cuda')
    
    position_ids = torch.arange(seq_length, dtype=torch.long, device='cuda').unsqueeze(0).expand(batch_size, -1)
    
    attention_mask = torch.tril(torch.ones((seq_length, seq_length), device='cuda', dtype=torch.bool))
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    labels = torch.cat([tokens[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long, device='cuda')], dim=1)
    
    return tokens, position_ids, attention_mask, labels


def custom_backend(gm: torch.fx.GraphModule, example_inputs: tuple[torch.Tensor, ...]) -> torch.fx.GraphModule:
    """Custom backend for megatron-core."""
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(gm.graph.python_code(root_module="self").src)
    return gm


torch._dynamo.config.compiled_autograd = True

@torch.compile(backend=custom_backend)
def training_step(model: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Perform one training step."""
    model.train()
    
    tokens, position_ids, attention_mask, labels = batch
    
    output = model(tokens, position_ids, attention_mask)
    
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, output.size(-1)), 
        labels.view(-1), 
        ignore_index=0
    )

    if TRAIN_MODE:
        loss.backward()
    
    return loss


def main() -> None:
    setup_distributed()
    model = create_model()
    optimizer = create_optimizer(model)

    batch = create_dummy_batch()
    torch._dynamo.mark_dynamic(batch[0], 0)
    torch._dynamo.mark_dynamic(batch[1], 0)
    torch._dynamo.mark_dynamic(batch[3], 0)
    training_step(model, batch)
    optimizer.step()
    optimizer.zero_grad()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
