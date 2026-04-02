"""
Inspect SmolVLA prefix token count and shapes.

Usage:
    python -m scripts.utils.inspect_prefix_tokens

No Isaac Sim required. Loads SmolVLA model, constructs dummy inputs,
runs embed_prefix, and prints per-component and total token counts.
"""

import sys
from pathlib import Path

import torch

# Add submission source to path so we can import the local SmolVLA
SUBMISSION_DIR = Path(__file__).resolve().parents[2] / "submission" / "source_code" / "lerobot_policies_smolvla"
sys.path.insert(0, str(SUBMISSION_DIR))

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, pad_vector


def main():
    # --- 1. Build config (matches your training config) ---
    config = SmolVLAConfig(
        # Image features: 3 cameras
        input_features={
            "observation.state": {"type": "STATE", "shape": [12]},
            "observation.images.top_rgb": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.left_rgb": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.images.right_rgb": {"type": "VISUAL", "shape": [3, 480, 640]},
        },
        output_features={"action": {"type": "ACTION", "shape": [12]}},
        device="cuda",
        n_action_steps=12,
        chunk_size=50,
        resize_imgs_with_padding=(512, 512),
        max_state_dim=32,
        max_action_dim=32,
        train_state_proj=False,
        freeze_vision_encoder=True,
        train_expert_only=True,
    )

    print(f"Config: max_state_dim={config.max_state_dim}, hidden_size expected=960")
    print(f"        resize_imgs={config.resize_imgs_with_padding}")
    print(f"        attention_mode={config.attention_mode}")
    print(f"        num_vlm_layers={config.num_vlm_layers}")
    print(f"        self_attn_every_n_layers={config.self_attn_every_n_layers}")
    print(f"        expert_width_multiplier={config.expert_width_multiplier}")
    print()

    # --- 2. Build model ---
    print("Loading SmolVLA model (VLM weights from HF cache)...")
    model = VLAFlowMatching(config)
    model.eval()

    vlm_hidden = model.vlm_with_expert.config.text_config.hidden_size
    expert_hidden = model.vlm_with_expert.expert_hidden_size
    print(f"VLM hidden_size = {vlm_hidden}")
    print(f"Expert hidden_size = {expert_hidden}")
    print()

    # --- 3. Prepare dummy inputs ---
    B = 1  # batch size

    # 3a. Images: 3 cameras, each (B, 3, 512, 512) normalized to [-1, 1]
    images = [
        torch.randn(B, 3, 512, 512),  # top_rgb
        torch.randn(B, 3, 512, 512),  # left_rgb
        torch.randn(B, 3, 512, 512),  # right_rgb
    ]
    img_masks = [torch.ones(B, dtype=torch.bool)] * 3

    # 3b. Language tokens: tokenize a sample instruction
    processor = model.vlm_with_expert.processor
    lang_text = "fold the garment"
    tokenized = processor.tokenizer(
        lang_text,
        return_tensors="pt",
        padding="max_length" if config.pad_language_to == "max_length" else True,
    )
    lang_tokens = tokenized["input_ids"].expand(B, -1)  # (B, ~30)
    lang_masks = tokenized["attention_mask"].expand(B, -1).bool()  # (B, ~30)

    # 3c. State: (B, 12) -> pad to max_state_dim=32
    state_raw = torch.randn(B, 12)
    state = pad_vector(state_raw, config.max_state_dim)  # (B, 32)

    # --- 4. Run embed_prefix and inspect each component ---
    print("=" * 60)
    print("Per-component token analysis:")
    print("=" * 60)

    with torch.no_grad():
        # 4a. Image embeddings - check each camera
        for cam_idx, (img, img_mask) in enumerate(zip(images, img_masks)):
            img_emb = model.vlm_with_expert.embed_image(img)
            print(f"  Camera {cam_idx}: img_emb shape = {img_emb.shape}  "
                  f"→ {img_emb.shape[1]} tokens × {img_emb.shape[2]}D")

        # 4b. Language embeddings
        lang_emb = model.vlm_with_expert.embed_language_tokens(lang_tokens)
        print(f"  Language:    lang_emb shape = {lang_emb.shape}  "
              f"→ {lang_emb.shape[1]} tokens × {lang_emb.shape[2]}D")

        # 4c. State embedding
        state_emb = model.state_proj(state)
        print(f"  State:       state_proj output = {state_emb.shape}  "
              f"→ 1 token × {state_emb.shape[-1]}D")

        # 4d. Full prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

    print()
    print("=" * 60)
    print("FINAL prefix_embs:")
    print(f"  shape = {prefix_embs.shape}")
    print(f"  → {prefix_embs.shape[1]} total tokens × {prefix_embs.shape[2]}D")
    print(f"  pad_masks shape  = {prefix_pad_masks.shape}")
    print(f"  att_masks shape  = {prefix_att_masks.shape}")
    print(f"  prefix_length    = {model.prefix_length}")
    print("=" * 60)

    # --- 5. Also run full prefix-only pass to confirm Expert no-op ---
    # IMPORTANT: use fill_kv_cache=True to match the real inference path
    # in sample_actions(). When fill_kv_cache=True, the condition at line 427
    # short-circuits, so ALL 16 layers use forward_attn_layer (self-attn),
    # never forward_cross_attn_layer. The even/odd alternating structure
    # does NOT apply during prefix pass.
    print()
    print("Running prefix-only VLM forward (fill_kv_cache=True, matches real inference)...")
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    with torch.no_grad():
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        outputs, kv_cache = model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # suffix=None → Expert no-op
            use_cache=True,
            fill_kv_cache=True,
        )

    # outputs is a list: [prefix_output, None]
    prefix_output = outputs[0]
    print(f"  prefix_output shape = {prefix_output.shape}")
    print(f"  Expert output       = {outputs[1]} (expected None)")
    print(f"  KV cache layers     = {len(kv_cache)} (one per VLM layer)")
    print()
    print(f"This is the z_{{1:M}} that RL Token Encoder receives:")
    print(f"   z = prefix_output of shape {prefix_output.shape}")
    print()
    print("Key findings:")
    print(f"  - 3 cameras x 64 tokens = 192 visual tokens")
    print(f"  - 3 language tokens + 1 state token = 4")
    print(f"  - Total: 196 tokens (compact!)")
    print(f"  - All 16 VLM layers do pure self-attn during prefix pass")
    print(f"  - Expert (hidden=720) is never involved")


if __name__ == "__main__":
    main()
