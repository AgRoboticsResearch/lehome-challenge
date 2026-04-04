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

    Saves ~30-80ms per decision chunk.
    """

    def __init__(
        self,
        pretrained_path: str | None = None,
        device: str = "cuda",
        task_description: str = "fold the garment",
        image_keys: list[str] | None = None,
        state_dim: int = 12,
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

    @torch.no_grad()
    def forward(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single VLM forward pass producing both z_vlm and the VLA action chunk.

        Args:
            obs_dict: Dict with image keys (observation.images.*_rgb) and
                      "observation.state" (joint positions).

        Returns:
            z_vlm:   (B, prefix_len, hidden_size) — VLM prefix hidden states for RL Token Encoder.
            a_tilde: (B, chunk_size, action_dim)  — VLA reference action chunk (unpadded).
        """
        # ── Phase 1: Prepare + embed prefix (same as VLAPrefixHook.extract_prefix) ──
        images, img_masks = self.prefix_hook.prepare_images(obs_dict)
        state = self.prefix_hook.prepare_state(obs_dict["observation.state"])
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

        # ── Phase 2: Single VLM forward — capture BOTH z_vlm and KV cache ──
        outputs_embeds, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        z_vlm = outputs_embeds[0]  # (B, ~196, 960)

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
