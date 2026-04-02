# RLT Implementation Plan: SmolVLA + RL Token Fine-Tuning

> Based on "RL Token: Bootstrapping Online RL with Vision-Language-Action Models" (Physical Intelligence)
> Adapted for SmolVLA (SmolVLM2-500M) + LeHome Challenge garment manipulation

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SmolVLA + RL Token Architecture                          │
│                                                                             │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐                                  │
│  │Images (x3)│ │Lang Token│ │State(12D)│                                  │
│  └─────┬─────┘ └────┬─────┘ └────┬─────┘                                  │
│        ▼            ▼            ▼                                          │
│  ┌────────────────────────────────────────────┐                            │
│  │           embed_prefix()                   │                            │
│  │  SigLIP -> [img_tok]                       │                            │
│  │  Text Embed -> [lang_tok]                  │                            │
│  │  state_proj -> [state_tok]                 │                            │
│  │  -> concat -> prefix_embs                  │                            │
│  └───────────────────┬────────────────────────┘                            │
│                      ▼                                                      │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │         SmolVLMWithExpertModel.forward()                     │         │
│  │                                                              │         │
│  │  VLM Backbone (SmolVLM2-500M, 16 Gemma2 layers)             │         │
│  │  Prefix pass: inputs_embeds=[prefix, None], fill_kv_cache=True│        │
│  │  -> 16 layers self-attn (Expert is no-op)                    │         │
│  │  -> final RMSNorm                                            │         │
│  │                                                              │         │
│  │  ┌──────────────────────────────────────────┐                │         │
│  │  │ outputs_embeds[0] = z_{1:M} [B, M, 960] │ <- fork point  │         │
│  │  └──────────┬───────────────┬───────────────┘                │         │
│  └─────────────┼───────────────┼────────────────────────────────┘         │
│                │               │                                           │
│  == PATH A: VLA Actions ==    == PATH B: RL Token ==                      │
│                │               │                                           │
│                ▼               ▼                                           │
│  ┌─────────────────────┐  ┌──────────────────────────────┐               │
│  │ 10-step ODE Denoise │  │ + Learnable token e_rl [1,960]│              │
│  │ (Expert cross-attn  │  │ -> [e_rl | z_{1:M}]           │              │
│  │  to VLM KV-cache)   │  │ -> 4-layer TransformerEncoder │              │
│  │                     │  │ -> z_rl at e_rl position      │              │
│  │ -> action_out_proj  │  │ -> Linear(960->256) projection │             │
│  │ -> a_tilde_{1:50}   │  │ -> z_rl [B, 256]              │              │
│  └──────────┬──────────┘  └──────────┬───────────────────┘               │
│             │                          │                                   │
│   Take first C=12 steps               │                                    │
│             │                          │                                    │
│             ▼                          ▼                                    │
│  ┌──────────────────────────────────────────────────┐                      │
│  │              RL Actor pi_theta                    │                      │
│  │  x = [z_rl(256), s_p(24)] = 280D                 │                      │
│  │  input = [x(280), a_tilde_flat(144)] = 424D      │                      │
│  │  MLP: 424 -> 512 -> 512 -> 144 (C x action_dim)  │                      │
│  │  Gaussian: mu_theta(x, a_tilde), fixed sigma      │                      │
│  │  Ref action dropout: 50% during training           │                      │
│  │  -> a_{1:12} [B, 12, 12]                          │                      │
│  └──────────────────────────────────────────────────┘                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────┐                      │
│  │          Twin Critic Q_psi                        │                      │
│  │  input = [x(280), a_flat(144)] = 424D            │                      │
│  │  Q1: 424 -> 512 -> 512 -> 1                       │                      │
│  │  Q2: 424 -> 512 -> 512 -> 1                       │                      │
│  │  min(Q1, Q2) for Bellman backup (TD3 style)       │                      │
│  └──────────────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Structural Design Decisions

### Decision A: Feature Extraction Point

| Item | Decision |
|------|----------|
| Extraction point | **After final RMSNorm** -- `outputs_embeds[0]` |
| Rationale | Consistent with pi0.6 semantics (standard transformer `last_hidden_state`); numerically stable; avoids scale drift from 16-layer residual accumulation |
| Code location | `smolvlm_with_expert.py:491-498` |
| What z_{1:M} contains | `[img_tokens \| lang_tokens \| state_token]` -- preserved in full, no slicing |

**Why Norm-after, not Norm-before:**

1. Standard transformer convention: `last_hidden_state` = post-RMSNorm output in all Gemma-based models
2. Scale stability: 16 layers of residual accumulation cause unpredictable hidden state norms; RMSNorm normalizes to consistent scale
3. pi0.6 equivalence: `z = f(s, l; theta_vla)` includes the complete VLM forward including final norm
4. Practical: no "double normalization" concern -- handle via RL Token Encoder input design

### Decision B: VLA Reference Action Generation

| Item | Decision |
|------|----------|
| Method | **Full 10-step ODE denoising** |
| Rationale | User confirmed compute budget is acceptable; reference action quality directly affects BC regularization quality |
| Code path | `VLAFlowMatching.sample_actions()` at `modeling_smolvla.py:794` |

**Compute breakdown per RL decision point:**

| Step | Compute | Frequency |
|------|---------|-----------|
| Prefix pass (1x) | ~500M params forward | Once |
| ODE denoise (10x) | ~200M params forward each | 10 times |
| RL Token Encoder | ~50M params forward | Once |
| Actor MLP | ~0.5M params forward | Once |

**z_rl extraction is essentially free** -- it reuses the prefix pass already computed for VLA action generation.

### Decision C: Chunk Alignment (VLA H=50 vs RL C=12)

| Item | Decision |
|------|----------|
| Actor input ref action | **Take first C=12 steps from VLA's 50-step output** |
| Remaining steps | **Discard, re-plan every C steps** |
| Rationale | Paper uses `a_tilde_{t:t+C-1}` not `a_tilde_{t:t+H-1}`; garment manipulation requires high-frequency closed-loop |

```
VLA output:  a_tilde_{1:50}
                 |
                 v
            Slice first 12
                 |
                 v
Actor input: a_tilde_{1:12}  -->  Actor output: a_{1:12}
                                          |
                                          v
                                    Execute 12 steps in env
                                          |
                                          v
                                    New obs -> re-plan
```

### Decision D: Environment Interaction Loop

| Item | Decision |
|------|----------|
| Transition format | **Chunk-level** -- one transition per C=12 steps |
| Execution within chunk | **Open-loop** -- z_rl and s_p fixed, execute a_{1:12} sequentially |
| Reward accumulation | **Discounted sum within chunk** per paper Eq.3 |
| Rationale | Dense reward makes sub-sampling unnecessary; chunk-level Q-value matches paper Eq.3 |

**Transition stored in replay buffer:**

```python
{
    "z_rl": z_rl_t0,           # [256] frozen RL token
    "s_p": s_p_t0,             # [24] joint_pos + joint_vel
    "ref_action": a_tilde,     # [12, 12] VLA reference (first C steps)
    "action": a,               # [12, 12] Actor output
    "reward": sum(gamma^t * r_t),  # discounted chunk return
    "next_z_rl": z_rl_t12,    # [256] next chunk's RL token
    "next_s_p": s_p_t12,      # [24] next chunk's proprioception
    "done": done_t12,          # terminal flag
}
```

**RL training loop pseudocode:**

```python
for episode in range(N_episodes):
    obs = env.reset()
    done = False
    while not done:
        # VLA forward -> a_tilde + z_rl
        with torch.no_grad():
            a_tilde_full = vla_policy.predict_action_chunk(obs)  # [1, 50, 12]
            a_tilde = a_tilde_full[:, :C, :]                     # [1, 12, 12]
            z_rl = rl_token_encoder(vlm_features)                 # [1, 256]

        # RL Actor -> a
        x = torch.cat([z_rl, s_p], dim=-1)   # [1, 280]
        a = actor.sample(x, a_tilde)           # [1, 12, 12]

        # Execute chunk, collect rewards
        rewards = []
        for t in range(C):
            obs, r, terminated, truncated, info = env.step(a[:, t, :])
            rewards.append(r)
            if terminated or truncated:
                done = True
                break

        # Chunk-level discounted return
        chunk_return = sum(gamma**t * r for t, r in enumerate(rewards))

        # Store transition
        replay_buffer.add(z_rl, s_p, a_tilde, a, chunk_return, next_z_rl, next_s_p, done)

        # Off-policy update
        for _ in range(G):  # G = update-to-data ratio
            batch = replay_buffer.sample()
            update_critic(batch)    # 2x per actor update
            update_actor(batch)     # 1x
```

### Decision E: RL Token Training Data Source

| Item | Decision |
|------|----------|
| Extraction path | **Prefix pass only** -- no need for ODE denoising |
| Training data | **Per garment type** from LeRobot demonstration dataset |
| z_{1:M} composition | **Full prefix** -- [img + lang + state] tokens, no slicing |
| Decoder target | **Post-Norm z_{1:M}** (same as encoder input) |

**Why prefix-pass-only is sufficient:**

```
Full VLA forward:
  1. embed_prefix(images, lang, state)     -> prefix_embs     [need this]
  2. vlm_with_expert.forward(prefix, None) -> z_{1:M} + KV    [need this]
  3. 10x ODE denoise (uses KV-cache)       -> a_tilde         [NOT needed]

Steps 1-2 are the prefix pass. Step 3 is the denoise pass.
z_{1:M} comes from step 2. Step 3 does not affect z_{1:M}.
```

**Stage 1 training pseudocode:**

```python
# Load frozen VLA
policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
policy.eval()
for p in policy.parameters():
    p.requires_grad = False

# Create trainable RL Token module
rl_token_module = RLTokenModule(hidden_dim=960, proj_dim=256)
optimizer = Adam(rl_token_module.parameters(), lr=1e-4)

# Iterate over demonstration dataset
dataset = LeRobotDataset(garment_type, root="Datasets/example/...")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for epoch in range(N_epochs):
    for batch in dataloader:
        with torch.no_grad():
            images, img_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]

            prefix_embs, prefix_pad_masks, prefix_att_masks = \
                policy.model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            outputs_embeds, _, _ = policy.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                inputs_embeds=[prefix_embs, None],
                fill_kv_cache=True,
            )
            z_vlm = outputs_embeds[0]  # [B, M, 960] post-RMSNorm

        z_rl = rl_token_module.encode(z_vlm)           # [B, 256]
        loss_ro = rl_token_module.decode_loss(z_rl, z_vlm)

        optimizer.zero_grad()
        loss_ro.backward()
        optimizer.step()
```

### Decision F: Proprioceptive State s_p

| Item | Decision |
|------|----------|
| s_p content | **[joint_pos(12), joint_vel(12)] = 24D** |
| Scope | **RL Actor input only** -- does not affect SmolVLA's 12D observation.state |
| Rationale | Paper uses "proprioceptive position and velocity"; velocity aids contact-rich garment manipulation |

**Why include velocity:**

1. Paper Appendix B explicitly uses position + velocity
2. Contact dynamics in garment manipulation depend on velocity (grasp force, drag speed)
3. Isaac Lab `Articulation.data.joint_vel` is readily available -- zero implementation cost
4. Actor input changes from 412D to 424D -- negligible dimension increase

**Implementation:** Add `joint_vel` to `_get_observations()` or extract directly from `Articulation.data.joint_vel` in the RL training loop.

**Note on redundancy with z_{1:M}:** z_{1:M} contains a state_token (joint_pos projected to 960D then processed by VLM). This provides high-level semantic understanding ("hand is near garment"). s_p provides raw numerical precision (shoulder_pan = 1.234 rad, velocity = 0.05 rad/s). These are complementary, not redundant -- analogous to vision + proprioception in biological systems.

### Decision G: Reward Strategy

| Item | Decision |
|------|----------|
| Reward source | **Existing `_get_rewards()` as-is** |
| Modifications | **None** -- no additional shaping |
| Rationale | Dense reward [0,1.0] is ~144x richer than paper's sparse binary signal |

**Existing reward properties:**

| Property | Value |
|----------|-------|
| Range | [0.0, 1.0] |
| Update frequency | Every 50 steps (~0.42s at 120Hz) |
| Intermediate steps | Cached from last computation |
| Composition | 80% primary (fold distance) + 20% secondary (shape) |
| Success bonus | 1.0 when all conditions met |
| Episode length | 7200 steps (60s at 120Hz) |

**Why no additional shaping is needed:**

| Shaping option | Rejected because |
|----------------|-----------------|
| Lower step_interval (e.g. 10) | 5x compute overhead; marginal benefit over already-dense reward |
| Action smoothness penalty | BC regularization `beta * ||a - a_tilde||^2` already constrains smoothness |
| Progress reward | Bellman backup `Q = r + gamma*V(s')` implicitly captures progress |
| Success bonus amplification | Critic's TD learning naturally amplifies terminal rewards |

## 3. File Structure

```
Modified files:
  submission/source_code/lerobot_policies_smolvla/
  ├── smolvlm_with_expert.py      # Expose vlm_features as 3rd return value
  └── modeling_smolvla.py          # Propagate vlm_features through VLAFlowMatching

New files:
  source/lehome/lehome/rl/
  ├── __init__.py
  ├── rl_token.py                  # RLTokenEncoder + RLTokenDecoder
  ├── actor.py                     # RLActor (Gaussian, conditioned on a_tilde)
  ├── critic.py                    # TwinCritic (TD3 style)
  ├── sac_trainer.py               # Off-policy trainer + ReplayBuffer
  └── vla_rl_policy.py             # Unified VLA-RL policy wrapper

  scripts/
  ├── train_rl_token.py            # Stage 1: offline RL Token training
  ├── train_rl_online.py           # Stage 2: online RL training
  └── eval_policy/
      └── rlt_policy.py            # Evaluation policy wrapper
```

## 4. Key Module Specifications

### RLTokenEncoder

```python
class RLTokenEncoder(nn.Module):
    """
    Input:  z_{1:M} [B, M, 960] (post-RMSNorm VLM features)
    Output: z_rl [B, 256] (compact RL token)

    Architecture:
    - Prepend 1 learnable token e_rl [1, 1, 960]
    - 4-layer TransformerEncoder (dim=960, heads=16)
    - Extract e_rl position output -> z_raw [B, 960]
    - Project: Linear(960 -> 256) -> z_rl [B, 256]
    """
    def __init__(self, hidden_dim=960, proj_dim=256, num_layers=4, num_heads=16):
        self.rl_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.encoder = nn.TransformerEncoder(...)
        self.projection = nn.Linear(hidden_dim, proj_dim)
```

### RLTokenDecoder

```python
class RLTokenDecoder(nn.Module):
    """
    Input:  z_rl [B, 256] + sg(z_{1:M}) [B, M, 960]
    Output: L_ro = sum ||h_phi(d_phi([z_rl, sg(z_1:i-1)]))_i - sg(z_i)||^2

    Autoregressive reconstruction. Forces z_rl to preserve task-relevant info.
    """
    def __init__(self, hidden_dim=960, proj_dim=256, num_layers=4, num_heads=16):
        self.inv_projection = nn.Linear(proj_dim, hidden_dim)
        self.decoder = nn.TransformerDecoder(...)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
```

### RLActor

```python
class RLActor(nn.Module):
    """
    pi_theta(a_{1:C} | x, a_tilde) = N(mu_theta(x, a_tilde), sigma^2 * I)
    x = [z_rl(256), s_p(24)] = 280D
    input = [x(280), a_tilde_flat(144)] = 424D
    MLP: 424 -> 512 -> 512 -> 144
    Reference action dropout: 50% during training
    """
    def __init__(self, rl_dim=256, state_dim=24, chunk=12, action_dim=12):
        input_dim = rl_dim + state_dim + chunk * action_dim  # 424
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, chunk * action_dim),
        )
        self.log_std = nn.Parameter(torch.full((chunk * action_dim,), -5.0))
```

### TwinCritic

```python
class TwinCritic(nn.Module):
    """
    Q_psi(x, a_{1:C}) -> scalar
    input = [x(280), a_flat(144)] = 424D
    Q1, Q2: 424 -> 512 -> 512 -> 1
    min(Q1, Q2) for Bellman backup
    """
    def __init__(self, rl_dim=256, state_dim=24, chunk=12, action_dim=12):
        input_dim = rl_dim + state_dim + chunk * action_dim  # 424
        self.q1 = QNetwork(input_dim, hidden=512)
        self.q2 = QNetwork(input_dim, hidden=512)
```

## 5. Code Modifications to Existing Files

### smolvlm_with_expert.py

**Line ~489** (after 16-layer loop, before final norm):

```python
# Save VLM raw features (pre-norm, but we extract post-norm below)
vlm_raw_features = inputs_embeds[0].clone()  # for backward compat if needed
```

**Line ~498** (modify return):

```python
# Before:
return outputs_embeds, past_key_values

# After:
return outputs_embeds, past_key_values, outputs_embeds[0]  # 3rd = z_{1:M} post-RMSNorm
```

**All callers** updated to accept 3rd return value. Backward compatible via optional parameter or unpacking.

### modeling_smolvla.py

**`VLAFlowMatching.forward()` (training, L756):** Capture vlm_features from vlm_with_expert.forward().

**`VLAFlowMatching.sample_actions()` (inference, L794):** Capture vlm_features from prefix pass; return alongside actions.

**`SmolVLAPolicy.forward()` and `SmolVLAPolicy.select_action()`:** Propagate vlm_features.

## 6. Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| RL Token count | 1 | Paper Fig.2 |
| z_rl dimension | 256 (projected from 960) | Design choice |
| RL chunk C | 12 | Matches n_action_steps |
| VLA chunk H | 50 | SmolVLA config |
| Actor hidden dim | 512 | Paper Appendix B |
| Critic hidden dim | 512 | Paper Appendix B |
| Actor layers | 3 (424->512->512->144) | Paper Appendix B |
| Critic layers | 3 per Q-network | Paper Appendix B |
| Encoder layers | 4 transformer layers | Design choice |
| Decoder layers | 4 transformer layers | Design choice |
| Update-to-data ratio G | 5 | Paper Section V |
| Critic/Actor update ratio | 2:1 | Paper Appendix B |
| Reference action dropout | 50% | Paper Section IV-B |
| BC regularization beta | 1.0 (tunable) | Paper Eq.5 |
| Discount factor gamma | 0.99 | Standard |
| Actor sigma (fixed std) | exp(-5) ~ 0.0067 | Paper Appendix B |
| RL Token training steps | 2000-10000 per garment type | Paper Appendix B |
| RL warmup episodes | N_warm (TBD) | Paper Algorithm 1 |
| RL total episodes | 400-1000 per garment type | Paper Appendix B |
| Optimizer | Adam | Standard |
| RL Token encoder lr | 1e-4 | Paper (implied) |
| Actor/Critic lr | 3e-4 (TBD) | Standard SAC |

## 7. Training Pipeline

```
Stage 1: RL Token Training (Offline)
─────────────────────────────────────
Input:  Frozen SmolVLA checkpoint + LeRobot demonstration data
Output: RL Token Encoder weights (per garment type)

  for garment_type in [top_long, top_short, pant_long, pant_short]:
    dataset = LeRobotDataset(garment_type)
    encoder = RLTokenEncoder(hidden_dim=960, proj_dim=256)
    decoder = RLTokenDecoder(hidden_dim=960)

    for step in range(2000-10000):
      batch = sample(dataset)
      z_vlm = prefix_pass(frozen_vla, batch)        # [B, M, 960]
      z_rl = encoder(z_vlm)                          # [B, 256]
      loss_ro = decoder.reconstruction_loss(z_rl, z_vlm)
      loss_ro.backward()
      optimizer.step()

    save("rl_token_encoder_{garment_type}.pt")


Stage 2: Online RL Training
────────────────────────────
Input:  Frozen VLA + Frozen RL Token Encoder + Isaac Sim environment
Output: Actor + Critic weights (per garment type)

  for garment_type in [top_long, top_short, pant_long, pant_short]:
    encoder = load("rl_token_encoder_{garment_type}.pt")
    actor = RLActor(rl_dim=256, state_dim=24, chunk=12, action_dim=12)
    critic = TwinCritic(rl_dim=256, state_dim=24, chunk=12, action_dim=12)
    buffer = ReplayBuffer()

    # Warmup
    for ep in range(N_warm):
      run_episode(vla_policy, env) -> fill buffer

    # Online RL
    for ep in range(400-1000):
      obs = env.reset()
      while not done:
        a_tilde, z_rl = vla_forward(frozen_vla, frozen_encoder, obs)
        a = actor.sample(z_rl, s_p, a_tilde)
        rewards = execute_chunk(env, a, C=12)
        buffer.add(z_rl, s_p, a_tilde, a, rewards, ...)
        for g in range(5):
          update_critic(critic, buffer)
          if g % 2 == 0:
            update_actor(actor, critic, buffer)

    save("rl_actor_{garment_type}.pt")
    save("rl_critic_{garment_type}.pt")
```

## 8. pi0.6 vs SmolVLA Architecture Comparison

| Aspect | pi0.6 (RLT paper) | SmolVLA (ours) | Impact on RLT |
|--------|-------------------|-----------------|----------------|
| VLM backbone | SigLIP(400M) + Gemma(4B) | SmolVLM2-500M (16 layers) | Smaller but sufficient |
| VLM hidden dim | 2048 | 960 | z_{1:M} dimension differs |
| Action Expert | 860M, separate module | 0.75x width (720), interleaved | Interleaved but isolated by attention mask |
| VLM-Expert compute | Serial (VLM then Expert) | Interleaved (same loop) | No impact on z_{1:M} extraction |
| Information flow | VLM -> Expert (one-way) | VLM -> Expert (one-way, same) | Equivalent for RLT |
| z_{1:M} content | Image tokens only (paper footnote) | [img + lang + state] tokens | Preserved in full; no slicing |
| Action dim | 14 | 12 (dual-arm) | Minor dimension adjustment |
| RL chunk C | 10 at 50Hz = 200ms | 12 at 120Hz = 100ms | Higher frequency control |
| Reward | Sparse binary (human-labeled) | Dense [0,1.0] (every ~0.4s) | **Major advantage** |

**Key architectural equivalence proof:**

In both pi0.6 and SmolVLA, the VLM branch never attends to Action Expert tokens (verified via attention mask analysis). This means:
- VLM outputs are identical regardless of whether Expert is present
- Prefix pass with `inputs_embeds=[prefix, None]` produces the same z_{1:M} as full forward
- Expert is no-op during prefix pass (`if hidden_states is None: continue`)
- RLT can be directly adapted without architectural conflicts

## 9. Environment Interface

| Property | Value |
|----------|-------|
| Env class | `GarmentEnv(DirectRLEnv)` at `garment_bi_v2.py:35` |
| Physics frequency | 120 Hz (`sim.dt = 1/120`) |
| Policy frequency | 120 Hz (`decimation = 1`) |
| Episode length | 7200 steps (60s) |
| Action space | 12D (6D per arm) |
| Observation | images (3x RGB) + joint_pos (12D) + depth |
| Reward | Dense [0,1.0] via `_get_rewards()` |
| Success check | `success_checker_garment_fold()` with `@step_interval(50)` |
| Done condition | Time-out only (`episode_length >= max`) |

**Available garment types:** `top_long`, `top_short`, `pant_long`, `pant_short`

**Available demonstration data:** `Datasets/example/{garment_type}_merged/` (LeRobot format)

## 10. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VLM features insufficient for RL Token | Low | SmolVLA already produces effective actions; features contain task-relevant info |
| RL Token encoder fails to converge | Medium | Monitor L_ro loss; increase training steps; try larger encoder |
| Actor diverges from VLA behavior | Medium | BC regularization beta; reference action dropout; tune beta upward |
| Reward too sparse at 50-step interval | Low | Can reduce step_interval to 10 if needed |
| Sim-to-real gap (future) | N/A (sim only) | N/A for current scope |
| Compute budget (CPU-only sim) | Medium | Profile VLA forward time; consider async rollout/update |

## 11. Implementation Order

| Phase | Description | Files | Est. Effort |
|-------|-------------|-------|-------------|
| 0 | Expose VLM features from SmolVLMWithExpertModel | `smolvlm_with_expert.py`, `modeling_smolvla.py` | Small |
| 1 | Implement RL Token Encoder + Decoder | `rl/rl_token.py` | Medium |
| 2 | Implement Actor + Critic | `rl/actor.py`, `rl/critic.py` | Medium |
| 3 | Stage 1 training script | `scripts/train_rl_token.py` | Medium |
| 4 | SAC trainer + replay buffer | `rl/sac_trainer.py` | Medium |
| 5 | Stage 2 online RL script | `scripts/train_rl_online.py` | Large |
| 6 | Evaluation policy wrapper | `scripts/eval_policy/rlt_policy.py` | Small |
