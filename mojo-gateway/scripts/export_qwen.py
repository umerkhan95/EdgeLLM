#!/usr/bin/env python3
"""
Export Qwen 2.5 to llama.c binary format for EdgeLLM inference.

Based on karpathy/llama2.c export.py
"""

import os
import sys
import struct
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def export_qwen(model_id: str, output_path: str):
    """Export Qwen model to llama.c binary format."""

    print(f"Loading model: {model_id}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    config = model.config

    # Extract config
    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    vocab_size = config.vocab_size
    seq_len = getattr(config, 'max_position_embeddings', 2048)

    print(f"Config:")
    print(f"  dim={dim}, hidden_dim={hidden_dim}")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, seq_len={seq_len}")

    head_size = dim // n_heads

    # Prepare output file
    print(f"\nExporting to: {output_path}")

    with open(output_path, 'wb') as f:
        # Write header (7 int32 values)
        # Format: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
        header = struct.pack('iiiiiii', dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        f.write(header)

        state_dict = model.state_dict()

        # Helper to write tensor
        def write_tensor(tensor, name=""):
            data = tensor.detach().cpu().float().numpy()
            f.write(data.tobytes())
            print(f"  {name}: {tensor.shape} -> {data.nbytes / 1024 / 1024:.2f} MB")

        # Token embeddings [vocab_size, dim]
        print("\nWriting embeddings...")
        write_tensor(state_dict['model.embed_tokens.weight'], 'embed_tokens')

        # RMS norm weights for attention
        print("\nWriting attention norms...")
        for i in range(n_layers):
            write_tensor(state_dict[f'model.layers.{i}.input_layernorm.weight'], f'layer{i}.attn_norm')

        # Attention weights
        print("\nWriting attention weights...")
        for i in range(n_layers):
            # Q projection [dim, dim]
            wq = state_dict[f'model.layers.{i}.self_attn.q_proj.weight']
            write_tensor(wq, f'layer{i}.wq')

            # K projection [dim, kv_dim]
            wk = state_dict[f'model.layers.{i}.self_attn.k_proj.weight']
            write_tensor(wk, f'layer{i}.wk')

            # V projection [dim, kv_dim]
            wv = state_dict[f'model.layers.{i}.self_attn.v_proj.weight']
            write_tensor(wv, f'layer{i}.wv')

            # Output projection [dim, dim]
            wo = state_dict[f'model.layers.{i}.self_attn.o_proj.weight']
            write_tensor(wo, f'layer{i}.wo')

        # RMS norm weights for FFN
        print("\nWriting FFN norms...")
        for i in range(n_layers):
            write_tensor(state_dict[f'model.layers.{i}.post_attention_layernorm.weight'], f'layer{i}.ffn_norm')

        # FFN weights (Qwen uses SwiGLU: gate_proj, up_proj, down_proj)
        print("\nWriting FFN weights...")
        for i in range(n_layers):
            # Gate projection (w1) [hidden_dim, dim]
            w1 = state_dict[f'model.layers.{i}.mlp.gate_proj.weight']
            write_tensor(w1, f'layer{i}.w1')

            # Down projection (w2) [dim, hidden_dim]
            w2 = state_dict[f'model.layers.{i}.mlp.down_proj.weight']
            write_tensor(w2, f'layer{i}.w2')

            # Up projection (w3) [hidden_dim, dim]
            w3 = state_dict[f'model.layers.{i}.mlp.up_proj.weight']
            write_tensor(w3, f'layer{i}.w3')

        # Final norm
        print("\nWriting final norm...")
        write_tensor(state_dict['model.norm.weight'], 'final_norm')

        # Output weights (lm_head) - may be tied to embeddings
        print("\nWriting output weights...")
        if 'lm_head.weight' in state_dict:
            write_tensor(state_dict['lm_head.weight'], 'lm_head')
        else:
            # Tied weights - write embeddings again
            write_tensor(state_dict['model.embed_tokens.weight'], 'lm_head (tied)')

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Output size: {file_size / 1024 / 1024:.2f} MB")

    # Export tokenizer
    tokenizer_path = output_path.replace('.bin', '_tokenizer.bin')
    print(f"\nExporting tokenizer to: {tokenizer_path}")
    export_tokenizer(tokenizer, tokenizer_path, vocab_size)

    return output_path, tokenizer_path


def export_tokenizer(tokenizer, output_path: str, vocab_size: int):
    """Export tokenizer to llama.c binary format."""

    with open(output_path, 'wb') as f:
        # Write vocab size and max token length
        max_token_length = 0
        tokens = []
        scores = []

        for i in range(vocab_size):
            try:
                token = tokenizer.decode([i])
                token_bytes = token.encode('utf-8')
            except:
                token_bytes = b'<unk>'

            tokens.append(token_bytes)
            scores.append(0.0)  # Qwen doesn't use scores like SentencePiece
            max_token_length = max(max_token_length, len(token_bytes))

        # Header
        f.write(struct.pack('i', vocab_size))
        f.write(struct.pack('i', max_token_length))

        # Write tokens
        for i, (token_bytes, score) in enumerate(zip(tokens, scores)):
            f.write(struct.pack('f', score))
            f.write(struct.pack('i', len(token_bytes)))
            f.write(token_bytes)

    print(f"Tokenizer exported: {os.path.getsize(output_path) / 1024:.2f} KB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Qwen model to llama.c format')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B',
                        help='HuggingFace model ID')
    parser.add_argument('--output', type=str, default='qwen2.5-1.5b.bin',
                        help='Output binary file')
    args = parser.parse_args()

    export_qwen(args.model, args.output)
