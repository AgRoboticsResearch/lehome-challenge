import json

import torch

from lehome.models.vla_prefix_hook import VLAPrefixHook
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


class VLAStage2Hook:
    """
    Unified VLA interface for Stage 2 RL: single VLM forward producing both z_vlm and a_tilde.

    Reuses VLAPrefixHook for model construction, weight loading, and data preprocessing.
    Adds a merged forward pass that captures both VLM hidden states (z_vlm) and the KV cache,
    then runs ODE denoising with the cached KV — avoiding the duplicate prefix encoding that
    separate VLAPrefixHook.extract_prefix() + sample_actions() would require.

    CRITICAL: Normalizes observation.state using dataset stats before feeding to the model,
    matching the eval pipeline's expert preprocessor behavior. Without this, the VLA receives
    raw joint positions instead of the normalized values it was trained on.

    Saves ~30-80ms per decision chunk.
    """

    def __init__(
        self,
        pretrained_path: str | None = None,
        device: str = "cuda",
        task_description: str = "fold the garment",
        image_keys: list[str] | None = None,
        state_dim: int = 12,
        dataset_stats_path: str | None = None,
    ):
        self.prefix_hook = VLAPrefixHook(
            pretrained_path=pretrained_path,
            device=device,
            task_description=task_description,
            image_keys=image_keys,
            state_dim=state_dim,
        )
        self.model = self.prefix_hook.model  # VLAFlowMatching, already frozen
        self.device = self.prefix_hook.device

        # Load state and action normalization stats (same as eval pipeline's expert preprocessor)
        self.state_mean = None
        self.state_std = None
        self.act_mean = None
        self.act_std = None
        if dataset_stats_path:
            with open(dataset_stats_path) as f:
                stats = json.load(f)
            self.state_mean = torch.tensor(
                stats["observation.state"]["mean"], device=device
            )
            self.state_std = torch.tensor(
                stats["observation.state"]["std"], device=device
            )
            self.act_mean = torch.tensor(
                stats["action"]["mean"], device=device
            )
            self.act_std = torch.tensor(
                stats["action"]["std"], device=device
            )

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state: (x - mean) / std, matching eval preprocessor."""
        if self.state_mean is not None:
            return (state - self.state_mean) / self.state_std
        return state

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Denormalize state: x * std + mean."""
        if self.state_std is not None:
            return state * self.state_std + self.state_mean
        return state

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action: (a - mean) / std."""
        if self.act_mean is not None:
            return (action - self.act_mean) / self.act_std
        return action

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action: a * std + mean."""
        if self.act_std is not None:
            return action * self.act_std + self.act_mean
        return action

    @torch.no_grad()
    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        skip_ode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Single VLM forward pass producing z_vlm and optionally the VLA action chunk.

        Args:
            obs_dict: Dict with image keys (observation.images.*_rgb) and
                      "observation.state" (joint positions, RAW or normalized).
            skip_ode: If True, skip ODE denoising and return a_tilde=None.
                      Saves ~30ms per chunk. Used when ref_action comes from MoE policy.

        Returns:
            z_vlm:   (B, prefix_len, hidden_size) — VLM prefix hidden states for RL Token Encoder.
            a_tilde: (B, chunk_size, action_dim)  — VLA reference action chunk (normalized space).
                     None if skip_ode=True.
        """
        # ── Phase 1: Prepare + embed prefix (same as VLAPrefixHook.extract_prefix) ──
        images, img_masks = self.prefix_hook.prepare_images(obs_dict)

        # Normalize state before feeding to model (CRITICAL: model was trained on normalized state)
        raw_state = obs_dict["observation.state"]
        if isinstance(raw_state, torch.Tensor):
            raw_state = raw_state.float().to(self.device)
        else:
            raw_state = torch.as_tensor(raw_state, dtype=torch.float32, device=self.device)
        state = self.normalize_state(raw_state)
        state = self.prefix_hook.prepare_state(state)

        B = state.shape[0]
        lang_tokens = self.prefix_hook._lang_tokens.expand(B, -1).to(self.device)
        lang_masks = self.prefix_hook._lang_masks.expand(B, -1).to(self.device)
        images = [img.to(self.device) for img in images]
        img_masks = [m.to(self.device) for m in img_masks]
        state = state.to(self.device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # ── Phase 2: VLM forward — z_vlm only when skip_ode ──
        # fill_kv_cache=True forces self-attn path (safe with inputs_embeds[1]=None).
        # Without it, cross-attn mode hits inputs_embeds[1] which is None → AttributeError.
        # When skip_ode=True, we still set use_cache=False so the KV cache is discarded.
        outputs_embeds, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=not skip_ode,
            fill_kv_cache=True,
        )
        z_vlm = outputs_embeds[0]  # (B, ~196, 960)

        if skip_ode:
            return z_vlm, None

        # ── Phase 3: ODE denoising using cached KV (no re-encoding) ──
        chunk_size = self.model.config.chunk_size  # 50
        max_action_dim = self.model.config.max_action_dim  # 32
        action_dim = self.prefix_hook.state_dim  # 12
        num_steps = self.model.config.num_steps  # 10
        dt = -1.0 / num_steps

        x_t = self.model.sample_noise((B, chunk_size, max_action_dim), self.device)
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=self.device).expand(B)
            v_t = self.model.denoise_step(
                prefix_pad_masks, past_key_values, x_t, time_tensor
            )
            x_t = x_t + dt * v_t

        a_tilde = x_t[:, :, :action_dim]  # (B, 50, 12)
        return z_vlm, a_tilde
