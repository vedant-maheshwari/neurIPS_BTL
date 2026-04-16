"""
SVD Attention Intervention for Phi-2
=====================================

Pipeline:
  1. Load Phi-2 and tokenize a prompt
  2. Run a baseline forward pass (original attention)
  3. Monkey-patch the LAST layer's attention to inject SVD truncation:
       - Compute attention weights as usual (QK^T / sqrt(d) + mask → softmax)
       - SVD decompose each head's attention matrix
       - Retain top-k singular modes, zero-pad the rest
       - Reconstruct the attention matrix
       - Use the reconstructed matrix to compute attn_output = A_reconstructed @ V
  4. Run the modified forward pass
  5. Compare baseline vs modified outputs (logits, decoded text, attention heatmaps)

Usage:
    python svd_attention_intervention.py \
        --prompt "Explain the concept of entropy in physics." \
        --top_k 5 \
        --max_new_tokens 50 \
        --device cpu
"""

import argparse
import math
import os
from functools import partial

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# SVD Truncation Utility
# ─────────────────────────────────────────────────────────────────────────────

def svd_truncate_attention(attn_weights: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Apply SVD truncation to attention weight matrices.

    Args:
        attn_weights: (batch, num_heads, seq_len, seq_len) — post-softmax attention
        top_k: number of singular modes to retain

    Returns:
        Reconstructed attention matrix with only the top-k modes.
    """
    B, H, S, S2 = attn_weights.shape
    assert S == S2, f"Attention matrix must be square, got {S}x{S2}"

    device = attn_weights.device
    dtype = attn_weights.dtype

    # Work in float32 for numerical stability during SVD
    attn_f32 = attn_weights.float()
    reconstructed = torch.zeros_like(attn_f32)

    for b in range(B):
        for h in range(H):
            A = attn_f32[b, h]  # (S, S)

            # SVD decomposition
            U, S_vals, Vt = torch.linalg.svd(A, full_matrices=False)
            # U: (S, S), S_vals: (S,), Vt: (S, S)

            # Retain top-k modes, zero the rest
            k = min(top_k, S_vals.shape[0])
            S_truncated = torch.zeros_like(S_vals)
            S_truncated[:k] = S_vals[:k]

            # Reconstruct: U @ diag(S_truncated) @ Vt
            reconstructed[b, h] = U @ torch.diag(S_truncated) @ Vt

    return reconstructed.to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Custom Attention Forward (with SVD injection)
# ─────────────────────────────────────────────────────────────────────────────

def svd_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    top_k: int = 5,
    capture_dict: dict = None,
    **kwargs,
):
    """
    Drop-in replacement for eager_attention_forward that injects SVD
    truncation on the attention weights before multiplying with values.
    """
    # Repeat KV heads if needed (for GQA; Phi-2 is MHA so n_rep=1)
    from transformers.models.phi.modeling_phi import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # Compute raw attention scores
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # Apply causal mask
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # ──── SVD TRUNCATION INJECTION ────
    original_attn_weights = attn_weights.clone()
    attn_weights = svd_truncate_attention(attn_weights, top_k=top_k)

    # Capture for analysis
    if capture_dict is not None:
        capture_dict["original_attn"] = original_attn_weights.detach().cpu()
        capture_dict["truncated_attn"] = attn_weights.detach().cpu()

    # Dropout (no-op during eval)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Weighted sum over values
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Monkey-Patching Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class SVDAttentionPatcher:
    """Context manager to patch the last layer's attention with SVD truncation."""

    def __init__(self, model, top_k: int = 5, layer_idx: int = -1):
        self.model = model
        self.top_k = top_k
        self.layer_idx = layer_idx
        self.capture = {}
        self._original_forward = None
        self._target_attn = None

    def __enter__(self):
        # Get the target layer's attention module
        layers = self.model.model.layers
        target_layer = layers[self.layer_idx]
        self._target_attn = target_layer.self_attn

        # Save original forward
        self._original_forward = self._target_attn.forward

        # Build the patched forward
        original_forward = self._original_forward
        top_k = self.top_k
        capture = self.capture
        target_attn = self._target_attn

        def patched_forward(
            hidden_states: torch.Tensor,
            position_embeddings: tuple,
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            **kwargs,
        ):
            """Patched attention forward that uses SVD-truncated attention."""
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, target_attn.head_dim)

            query_states = target_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = target_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = target_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            if target_attn.qk_layernorm:
                query_states = target_attn.q_layernorm(query_states)
                key_states = target_attn.k_layernorm(key_states)

            cos, sin = position_embeddings

            # Partial rotary embedding
            from transformers.models.phi.modeling_phi import apply_rotary_pos_emb
            query_rot, query_pass = (
                query_states[..., : target_attn.rotary_ndims],
                query_states[..., target_attn.rotary_ndims :],
            )
            key_rot, key_pass = (
                key_states[..., : target_attn.rotary_ndims],
                key_states[..., target_attn.rotary_ndims :],
            )
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

            query_states = torch.cat((query_rot, query_pass), dim=-1)
            key_states = torch.cat((key_rot, key_pass), dim=-1)

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, target_attn.layer_idx
                )

            # Use SVD-modified attention instead of the standard one
            attn_output, attn_weights = svd_eager_attention_forward(
                target_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0,
                scaling=target_attn.scaling,
                top_k=top_k,
                capture_dict=capture,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = target_attn.dense(attn_output)
            return attn_output, attn_weights

        # Apply the patch
        self._target_attn.forward = patched_forward
        return self

    def __exit__(self, *args):
        # Restore original forward
        self._target_attn.forward = self._original_forward


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_comparison(capture: dict, head_idx: int, tokens: list[str], save_path: str):
    """Plot original vs SVD-truncated attention matrices side by side."""
    orig = capture["original_attn"][0, head_idx].numpy()
    trunc = capture["truncated_attn"][0, head_idx].numpy()
    diff = orig - trunc

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Truncate token labels for readability
    labels = [t[:12] for t in tokens]
    n = len(labels)

    for ax, data, title in zip(
        axes,
        [orig, trunc, diff],
        ["Original Attention", "SVD-Truncated Attention", "Difference (Orig − Truncated)"],
    ):
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_title(title, fontsize=13, fontweight="bold")
        if n <= 30:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Head {head_idx} — Attention Matrix SVD Intervention", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved attention comparison plot → {save_path}")


def plot_singular_value_spectrum(capture: dict, save_path: str):
    """Plot singular value spectra for all heads (original attention)."""
    orig = capture["original_attn"][0]  # (H, S, S)
    H = orig.shape[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab20", H)

    for h in range(H):
        A = orig[h].numpy()
        sv = np.linalg.svd(A, compute_uv=False)
        ax.plot(sv, color=cmap(h), alpha=0.7, label=f"Head {h}")

    ax.set_xlabel("Singular Value Index", fontsize=12)
    ax.set_ylabel("Singular Value Magnitude", fontsize=12)
    ax.set_title("Singular Value Spectrum — Last Layer Attention (All Heads)", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved singular value spectrum → {save_path}")


def plot_energy_retention(capture: dict, top_k: int, save_path: str):
    """Bar chart showing energy retained per head after SVD truncation."""
    orig = capture["original_attn"][0]  # (H, S, S)
    H = orig.shape[0]

    energy_retained = []
    for h in range(H):
        A = orig[h].numpy()
        sv = np.linalg.svd(A, compute_uv=False)
        total_energy = np.sum(sv**2)
        k = min(top_k, len(sv))
        retained_energy = np.sum(sv[:k]**2)
        energy_retained.append(retained_energy / total_energy * 100)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(H), energy_retained, color=plt.cm.viridis(np.array(energy_retained) / 100), edgecolor="black", linewidth=0.5)
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="100%")
    ax.set_xlabel("Head Index", fontsize=12)
    ax.set_ylabel("Energy Retained (%)", fontsize=12)
    ax.set_title(f"Energy Retained per Head (top-{top_k} modes)", fontsize=14, fontweight="bold")
    ax.set_xticks(range(H))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    # Annotate bars
    for bar, pct in zip(bars, energy_retained):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Saved energy retention plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SVD Attention Intervention on Phi-2")
    parser.add_argument("--prompt", type=str,
                        default="Explain the concept of entropy in thermodynamics step by step.",
                        help="Input prompt")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top singular modes to retain")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Max tokens to generate for text comparison")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to run on")
    parser.add_argument("--output_dir", type=str, default="./svd_outputs",
                        help="Directory to save plots")
    parser.add_argument("--layer_idx", type=int, default=-1,
                        help="Which layer to intervene on (default: last layer, -1)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load Model & Tokenizer ──
    print("=" * 70)
    print("  SVD Attention Intervention — Phi-2")
    print("=" * 70)
    print(f"\n🔧 Config:")
    print(f"   Prompt:          {args.prompt[:60]}...")
    print(f"   Top-k modes:     {args.top_k}")
    print(f"   Layer index:     {args.layer_idx}")
    print(f"   Device:          {args.device}")
    print(f"   Max new tokens:  {args.max_new_tokens}")
    print()

    print("📦 Loading Phi-2 model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",  # Force eager attention for hook compatibility
    ).to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(model.model.layers)
    actual_layer = args.layer_idx if args.layer_idx >= 0 else num_layers + args.layer_idx
    print(f"   Model loaded. {num_layers} layers, intervening on layer {actual_layer}")
    print()

    # ── 2. Tokenize ──
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    input_ids = inputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = input_ids.shape[1]
    print(f"📝 Prompt tokenized: {seq_len} tokens")
    print(f"   Tokens: {tokens}")
    print()

    # ── 3. Baseline Forward Pass ──
    print("🔵 Running BASELINE forward pass (original attention)...")
    with torch.no_grad():
        baseline_outputs = model(**inputs)
    baseline_logits = baseline_outputs.logits
    baseline_next_token_logits = baseline_logits[0, -1, :]
    baseline_top5 = torch.topk(baseline_next_token_logits, 5)
    print("   Top-5 next token predictions (baseline):")
    for i, (idx, logit) in enumerate(zip(baseline_top5.indices, baseline_top5.values)):
        token = tokenizer.decode(idx)
        print(f"     {i+1}. '{token}' (logit={logit.item():.4f})")
    print()

    # ── 4. SVD-Modified Forward Pass ──
    print(f"🟡 Running SVD-MODIFIED forward pass (top-{args.top_k} modes)...")
    with torch.no_grad():
        with SVDAttentionPatcher(model, top_k=args.top_k, layer_idx=args.layer_idx) as patcher:
            modified_outputs = model(**inputs)
            capture = patcher.capture

    modified_logits = modified_outputs.logits
    modified_next_token_logits = modified_logits[0, -1, :]
    modified_top5 = torch.topk(modified_next_token_logits, 5)
    print("   Top-5 next token predictions (SVD-modified):")
    for i, (idx, logit) in enumerate(zip(modified_top5.indices, modified_top5.values)):
        token = tokenizer.decode(idx)
        print(f"     {i+1}. '{token}' (logit={logit.item():.4f})")
    print()

    # ── 5. Comparison Metrics ──
    print("📊 Logit Comparison:")
    logit_diff = (baseline_logits - modified_logits).abs()
    print(f"   Mean absolute logit difference:  {logit_diff.mean().item():.6f}")
    print(f"   Max absolute logit difference:   {logit_diff.max().item():.6f}")
    print(f"   L2 norm of logit difference:     {logit_diff.norm().item():.6f}")

    # KL divergence between output distributions
    baseline_probs = torch.softmax(baseline_next_token_logits, dim=0)
    modified_probs = torch.softmax(modified_next_token_logits, dim=0)
    kl_div = torch.sum(baseline_probs * (torch.log(baseline_probs + 1e-10) - torch.log(modified_probs + 1e-10)))
    print(f"   KL divergence (baseline || modified): {kl_div.item():.6f}")
    print()

    # ── 6. Text Generation Comparison ──
    print("📝 Text Generation Comparison:")
    print("   Generating with BASELINE model...")
    with torch.no_grad():
        baseline_gen = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    baseline_text = tokenizer.decode(baseline_gen[0], skip_special_tokens=True)
    print(f"   Baseline: {baseline_text}")
    print()

    print(f"   Generating with SVD-MODIFIED model (top-{args.top_k})...")
    with torch.no_grad():
        with SVDAttentionPatcher(model, top_k=args.top_k, layer_idx=args.layer_idx):
            modified_gen = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
    modified_text = tokenizer.decode(modified_gen[0], skip_special_tokens=True)
    print(f"   Modified: {modified_text}")
    print()

    # ── 7. Visualization ──
    if "original_attn" in capture:
        print("🎨 Generating visualizations...")
        num_heads = capture["original_attn"].shape[1]

        # Singular value spectrum
        plot_singular_value_spectrum(capture, os.path.join(args.output_dir, "sv_spectrum.png"))

        # Energy retention
        plot_energy_retention(capture, args.top_k, os.path.join(args.output_dir, "energy_retention.png"))

        # Attention comparison for first 4 heads
        for h in range(min(4, num_heads)):
            plot_attention_comparison(
                capture, h, tokens,
                os.path.join(args.output_dir, f"attn_comparison_head{h}.png"),
            )

    # ── 8. Summary ──
    print()
    print("=" * 70)
    print("  ✅ SVD Attention Intervention Complete")
    print("=" * 70)
    print(f"  Outputs saved to: {args.output_dir}/")
    print(f"  Top-k modes retained: {args.top_k}")
    print(f"  Layer intervened: {actual_layer} (of {num_layers})")
    if "original_attn" in capture:
        print(f"  Attention matrix shape: {capture['original_attn'].shape}")
    print()


if __name__ == "__main__":
    main()
