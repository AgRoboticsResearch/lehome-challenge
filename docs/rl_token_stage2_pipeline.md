# RL Token Stage 2: Online RL with Actor-Critic — Implementation Plan

## Context

Stage 1 (offline, already implemented) trains an RL Token Encoder/Decoder to compress SmolVLA's 196-token prefix into a single 960D z_rl. Stage 2 uses this frozen z_rl as the state representation for lightweight online RL, training a small Actor-Critic to refine the VLA's action chunks in the Isaac Sim garment folding environment.

**Paper**: "RL Token: Bootstrapping Online RL with VLA Models" (Physical Intelligence, 2025)

---

## Key Design Decisions (confirmed with user)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Device** | **Policy on GPU, sim on CPU** | CPU only for garment physics; VLA/actor/critic all on GPU |
| Action chunk C | **10** | Paper uses C=10; at ~30Hz sim = 0.33s per chunk |
| VLA reference | **Full VLA on GPU every C steps** | ~50-200ms on GPU (vs 2-5s on CPU). Paper-faithful. |
| Actor/Critic state | **[z_rl(960) + joint_pos(12)] = 972D** | Symmetric: actor & critic share same state input |
| RL algorithm | **TD3+BC** (twin critic, fixed sigma, BC regularization) | Paper Section IV-B |
| Reward | **Dense reward** (existing 0-1 from env) | ~144x richer signal than paper's sparse +1/0 |
| Reference dropout | **50%** during training, always provided at eval | Paper Section IV-B |
| Eval ã | **Full VLA at eval** | Paper-faithful, fast enough on GPU |
| **Normalization** | **Dataset stats only** | joint_pos/action 用 dataset stats.json；z_rl 不归一化（已由 Stage 1 控制） |
| **Actor 初始化** | **BC pretrain from warmup data** | 从 VLA 行为克隆开始，避免 random actor 产生垃圾数据 |
| **VLA 集成** | **VLAStage2Hook（共享 KV Cache）** | 一次 VLM forward 同时产出 z_vlm 和 ã，避免重复计算，每 chunk 省 ~60ms |
| **next_z_rl** | **延迟一拍存储** | 用当前 chunk 的 z_rl 作为上一条 transition 的 next_z_rl，每 chunk 只需一次 VLA forward |
| **Garment 范围** | **单 garment** | 先只支持单 garment 类型（如 top_long），后续再扩展 |

---

## Architecture Overview

```
Device split:
  GPU: VLA inference, RL Token encoder, Actor, Critic, ReplayBuffer, RL updates
  CPU: Isaac Sim garment physics (env.step())

Every C=10 steps:

1. OBSERVE (CPU → GPU transfer)
   obs = env._get_observations()   # CPU numpy dicts
   obs_gpu = {k: torch.as_tensor(v).to(device) for k,v in obs.items()}

2. VLA UNIFIED FORWARD on GPU (frozen, ~110ms, shared KV cache)
   z_vlm, a_tilde = vla_hook.forward(obs_gpu)
                     ↑ z_vlm: (1, 196, 960)    ← VLM hidden states
                     ↑ a_tilde: (1, 50, 12)    ← ODE denoised actions
                     ↑ 一次 VLM forward，KV cache 在 z_vlm 提取和 ã 去噪间共享

3. RL TOKEN on GPU (frozen, fast)
   z_target = apply_keep_mask(z_vlm)                     # (1, 193, 960)
   z_rl = rl_token_encoder(z_target)                     # (1, 960)

4. ASSEMBLE + NORMALIZE STATES
   joint_pos_norm = (joint_pos - mean) / std             # dataset stats
   state = [z_rl(960), joint_pos_norm(12)]               # (972,) for BOTH actor & critic

5. ACTOR on GPU (trainable, ~1ms)
   50% dropout: ref = ã OR zeros
   action_chunk = actor(state, ref)                       # (1, 10, 12) normalized space
   action_raw = action_chunk * act_std + act_mean         # denormalize to joint-space
   action_cpu = action_raw.cpu().numpy()

6. EXECUTE open-loop on CPU (C=10 sim steps)
   rewards = []
   for t in range(C):
     obs_t, reward_t, done_t = env.step(action_cpu[t])
     rewards.append(reward_t)
     if done_t:
       break                          # early termination on episode boundary
   n_exec = len(rewards)
   chunk_return = Σ_{t=0}^{n_exec-1} γ^t * r_t

7. STORE in replay buffer (on GPU)
   if n_exec < C:                              # partial chunk (episode ended mid-chunk)
     action_stored = action_chunk.clone()
     action_stored[:, n_exec:] = 0              # zero-pad unexecuted actions
   else:
     action_stored = action_chunk

   if done:
     next_z_rl = zeros, next_s_p = zeros        # terminal state
   else:
     next_z_rl, next_s_p = process_observation(obs_t, ...)

   (z_rl, s_p, ref, action_stored, chunk_return, next_z_rl, next_s_p, done)

8. OFF-POLICY UPDATE on GPU (G=5 iterations)
   batch = replay_buffer.sample(256)
   2x critic_update → 1x actor_update → target soft update
```

---

## VLA Unified Hook (VLAStage2Hook)

### 问题：为什么不能分开调用

Stage 2 每个chunk需要**同时**获取两个输出：
- **z_vlm** (196×960) → 送入 RL Token Encoder → z_rl（状态表示）
- **ã** (50×12) → VLA 参考动作（Actor 的 ref 输入 + BC 正则化 target）

SmolVLA 的 `sample_actions()` 内部是严格的两阶段 pipeline：

```
Phase 1: VLM Forward（昂贵，~50-80ms）
  embed_prefix(images, lang, state) → prefix_embs (~778 tokens)
  vlm_with_expert.forward(prefix_embs, fill_kv_cache=True)
    → 返回两个值:
      (1) outputs_embeds[0] = z_vlm (hidden states)
      (2) past_key_values    = KV cache (用于 Phase 2)

Phase 2: ODE Denoising（~20-50ms，10步）
  for step in range(10):
    denoise_step(x_t, t, past_key_values)  # 复用 KV cache
    x_t += dt * v_t
  return x_t → ã (50×12)
```

现有代码的问题：

| 组件 | 调用了什么 | 保留了什么 | **丢弃了什么** |
|---|---|---|---|
| `VLAPrefixHook.extract_prefix()` | Phase 1 完整 VLM forward | `outputs[0]` (z_vlm) ✅ | `past_key_values` ❌ |
| `VLAFlowMatching.sample_actions()` | Phase 1 + Phase 2 | `past_key_values` → ã ✅ | `outputs[0]` (z_vlm) ❌ |

两者各做一次完整的 VLM forward，各丢弃对方需要的输出。**分开调用 = VLM backbone 跑两次 = ~2x 耗时。**

### 什么是 KV Cache

Transformer 的 Self-Attention 为每个 token 计算 Q (Query)、K (Key)、V (Value)。
当序列分为 Prefix（固定的观测）和 Suffix（变化的 action tokens）时：

- Prefix tokens 的 K 和 V 在整个 action 生成过程中**不变**
- KV Cache = 把 Prefix 的 K/V 存起来，10 步 ODE denoising 直接复用，不重复计算

```
没有 KV Cache:
  去噪第1步: 算 Prefix(778 tokens) 的 K,V + Suffix(50 tokens) 的 K,V → Attention
  去噪第2步: 算 Prefix(778 tokens) 的 K,V + Suffix(50 tokens) 的 K,V → Attention
  ...（重复 10 次）
  总 token 处理量: 778 + 10×828 = 9,058

有 KV Cache:
  预先算一次: Prefix 的 K,V → 存入 cache
  去噪第1步: 取缓存的 K,V + 只算 Suffix(50) 的 K,V → Attention
  去噪第2步: 取缓存的 K,V + 只算 Suffix(50) 的 K,V → Attention
  ...（复用 10 次）
  总 token 处理量: 778 + 10×50 = 1,278

加速比: ~7x
```

### 解决方案：VLAStage2Hook

一次 VLM forward 同时产出 z_vlm 和 ã，共享 KV Cache：

```python
class VLAStage2Hook:
    """
    统一的 VLA 接口：一次 VLM forward 同时返回 z_vlm 和 ã。
    
    与现有 VLAPrefixHook 的关系：
    - 复用 VLAPrefixHook 的 __init__（构建模型、加载权重、tokenize 语言）
    - 复用 prepare_images()、prepare_state() 等数据预处理方法
    - 额外调用 VLAFlowMatching 的 denoise_step()（ODE 去噪生成 ã）
    
    不比现有 VLAPrefixHook 更 "hacky"——访问的是同层级的内部 API。
    """
    
    def __init__(self, pretrained_path, device, task_description, image_keys, state_dim=12):
        # 复用 VLAPrefixHook 的初始化逻辑
        self.prefix_hook = VLAPrefixHook(
            pretrained_path=pretrained_path,
            device=device,
            task_description=task_description,
            image_keys=image_keys,
            state_dim=state_dim,
        )
        self.model = self.prefix_hook.model  # VLAFlowMatching, 已冻结
        self.device = self.prefix_hook.device
    
    @torch.no_grad()
    def forward(self, obs_dict):
        """
        一次 VLM forward 同时产出 z_vlm 和 ã。
        
        Args:
            obs_dict: 包含 observation.state, observation.images.* 的 dict
        
        Returns:
            z_vlm: (B, ~196, 960) — VLM hidden states，送入 RL Token Encoder
            a_tilde: (B, 50, 12) — VLA 参考动作，送入 Actor 作为 ref
        """
        # ── Phase 1: Embed Prefix（与 VLAPrefixHook.extract_prefix() 相同）──
        images, img_masks = self.prefix_hook.prepare_images(obs_dict)
        state = self.prefix_hook.prepare_state(obs_dict["observation.state"])
        B = state.shape[0]
        lang_tokens = self.prefix_hook._lang_tokens.expand(B, -1).to(self.device)
        lang_masks = self.prefix_hook._lang_masks.expand(B, -1).to(self.device)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # ── 一次 VLM forward，同时获取 z_vlm 和 KV cache ──
        outputs_embeds, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,           # ← 关键：保留 KV cache
            fill_kv_cache=True,
        )
        z_vlm = outputs_embeds[0]  # (B, ~196, 960)
        
        # ── Phase 2: ODE Denoising（复用 KV cache，不重复 VLM forward）──
        chunk_size = self.model.config.chunk_size  # 50
        max_action_dim = self.model.config.max_action_dim  # 32
        action_dim = 12
        num_steps = self.model.config.num_steps  # 10
        dt = -1.0 / num_steps
        
        x_t = self.model.sample_noise((B, chunk_size, max_action_dim), self.device)
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, device=self.device).expand(B)
            v_t = self.model.denoise_step(
                prefix_pad_masks, past_key_values, x_t, time_tensor
            )
            x_t = x_t + dt * v_t
        
        a_tilde = x_t[:, :, :action_dim]  # (B, 50, 12)
        return z_vlm, a_tilde
```

### 性能对比（实测，GPU warmup 后 5 次平均）

| 方案 | VLM Forward 次数 | 每chunk耗时 | 500 episodes |
|---|---|---|---|
| 分开调用（VLAPrefixHook + sample_actions） | 2 次 | ~147ms | ~7.4 小时 |
| **统一 Hook（VLAStage2Hook）** | **1 次** | **~110ms** | **~5.5 小时** |
| 节省 | 1 次 prefix | **~37ms (25%)** | **~1.9 小时** |
| 节省 | 50% | ~37% | **~3.6 小时** |

---

## Normalization Strategy

### Why needed

Stage 2 的 Actor/Critic 都是 MLP (512→512)，对输入尺度敏感。实际数据分布：

| 输入 | 范围 | std 跨维度差异 | 方案 |
|------|------|----------------|------|
| joint_pos (12D) | [-1.73, 1.65] | 0.16 ~ 0.94 (6x) | dataset stats.json |
| action (120D) | 和 joint_pos 同空间 | 同上 | dataset stats.json |
| z_rl (960D) | norm ≈ 27, element ≈ 0.87 | 和 joint 差 1-3x | **不 normalize**（Stage 1 已控制） |

### 为什么 z_rl 不需要 normalize

- z_rl 是 Stage 1 encoder 的输出，已经过 2 层 transformer + LayerNorm
- Stage 1 训练时 z_rl_norm 稳定在 ~27，分布已经比较规整
- 如果强行 normalize，训练和 eval 需要维护额外的 stats，增加复杂度
- MLP 第1层本身可以学到对 z_rl 的线性缩放（等价于 affine normalize）

### Dataset-only normalization

```
                ┌───────────────────────────────────────────┐
                │       Normalization: Dataset Only          │
                │            (stats.json)                    │
                ├───────────────────────────────────────────┤
                │ joint_pos (12D)  → (x - mean) / std      │
                │ action (120D)    → (x - mean) / std      │
                │ ref ã (120D)     → already normalized     │
                │ z_rl (960D)      → NO normalization       │
                └───────────────────────────────────────────┘
```

**数据流：**

```
env (raw space)                              actor/critic
──────────────                              ──────────────────────────────────
raw joint_pos ──── normalize(dataset_stats) ──→ state (12D, normalized)
z_rl ────────────────────────────────────────→ state (960D, as-is from Stage 1)
VLA ã ───────────────────────────────────────→ ref (already normalized space)
actor output (normalized) ── denormalize ────→ env.step(raw action)
BC loss: ||a_norm - ã_norm||²  (naturally aligned!)
```

### SimpleNormalizer class

```python
class SimpleNormalizer:
    """
    Normalizer using only dataset stats.json.
    No warmup, no running stats — simple and deterministic.

    Stats source: Datasets/example/top_long_merged/meta/stats.json
    Contains: observation.state (mean, std), action (mean, std)
    """

    def __init__(self, stats_path: str, device: torch.device):
        stats = json.load(open(stats_path))
        self.pos_mean = torch.tensor(stats["observation.state"]["mean"], device=device)
        self.pos_std = torch.tensor(stats["observation.state"]["std"], device=device)
        self.act_mean = torch.tensor(stats["action"]["mean"], device=device)
        self.act_std = torch.tensor(stats["action"]["std"], device=device)

    # --- Normalize (raw → normalized) ---
    def normalize_state(self, joint_pos):
        """joint_pos raw → normalized (12D)"""
        return (joint_pos - self.pos_mean) / self.pos_std

    def normalize_action(self, action):
        """action raw → normalized (12D)"""
        return (action - self.act_mean) / self.act_std

    # --- Denormalize (normalized → raw) ---
    def denormalize_action(self, action):
        """actor output (normalized) → raw action for env"""
        return action * self.act_std + self.act_mean

    def denormalize_state(self, state):
        """normalized state → raw joint_pos"""
        return state * self.pos_std + self.pos_mean
```

### Key invariant

> **Replay buffer stores joint_pos in normalized space, z_rl in raw space, actions in normalized space.**
> BC loss `||a - ã||²` compares normalized actions, naturally aligned because VLA outputs
> are also in normalized space. z_rl is stored as-is (no normalization).

### Episode boundary handling (partial chunks)

When episode ends mid-chunk (e.g. C=10 but done=True at step 5):

```
Scenario: C=10, episode ends at step 5

Execute:  steps 0-4 only (5 of 10)
  for t in range(C):
    obs, r, d = env.step(action[t])
    rewards.append(r)
    if d: break                                # stop immediately

Store:
  n_exec = 5                                   # actual steps executed
  chunk_return = Σ_{t=0}^{4} γ^t * r_t        # only 5 terms
  action_stored[:, 5:] = 0                      # zero-pad phantom steps
  done = True
  next_z_rl = zeros                             # terminal → zero next state
  next_s_p = zeros

Why this is safe:
  TD target = chunk_return + γ^C * (1-done) * Q_target(s', a')
                     = chunk_return + 0              ← (1-done) = 0 when done=True
                     = chunk_return                 ← phantom actions never affect learning

  The zero-padded phantom actions reduce critic variance when done=True,
  though mathematically any value would work since the bootstrapping term is zeroed.
```

---

## Files to Create

### 2. `scripts/train_rl_token_stage2.py` (~450 lines)
### 3. `scripts/eval_policy/rlt_policy.py` (~150 lines)
### 4. `scripts/eval_policy/__init__.py` (+1 行)Core Stage 2 components:

#### ReplayBuffer

Circular buffer storing chunk-level transitions on GPU:

```
Fields (each a pre-allocated tensor):
  z_rl:          (capacity, 960)       # raw (no normalization)
  s_p:           (capacity, 12)        # normalized joint_pos via dataset stats
  ref_action:    (capacity, 10, 12)    # VLA ã (already normalized)
  action:        (capacity, 10, 12)    # actor output (normalized space)
  reward:        (capacity,)           # raw discounted chunk return
  next_z_rl:     (capacity, 960)       # raw
  next_s_p:      (capacity, 12)        # normalized
  done:          (capacity,)           # bool

Methods:
  add(z_rl, s_p, ref, action, reward, next_z_rl, next_s_p, done)
  sample(batch_size) → dict of tensors
  __len__() → int
```

#### RLActor

Gaussian policy with reference conditioning:

```python
class RLActor(nn.Module):
    """
    Input:  [z_rl(960) + joint_pos_norm(12)] + [ref_action_flat(120)] = 1092
    MLP:    1092 → 512 → ReLU → 512 → ReLU → 120
    Output: mu(10,12), fixed σ = exp(-5) ≈ 0.0067
    """

    def __init__(self, z_rl_dim=960, state_dim=12, chunk_size=10, action_dim=12,
                 hidden_dim=512, num_layers=3, fixed_std=0.0067, ref_dropout=0.5):
        x_dim = z_rl_dim + state_dim            # 972
        ref_dim = chunk_size * action_dim         # 120
        input_dim = x_dim + ref_dim               # 1092
        output_dim = chunk_size * action_dim       # 120

        # 3-layer MLP: 1092 → 512 → 512 → 120
        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.ReLU()])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_rl, s_p, ref_action) -> action:
        """Training forward: reparameterization a = mu + σ·ε"""

    def get_deterministic_action(self, z_rl, s_p, ref_action) -> action:
        """Eval forward: return mu without noise"""

    def apply_ref_dropout(self, ref_action, training=True) -> ref_action:
        """50% dropout: replace with zeros"""
```

#### TwinCritic

Ensemble of 2 Q-functions (symmetric, same state as actor):

```python
class TwinCritic(nn.Module):
    """
    Input:  [z_rl(960) + joint_pos_norm(12)] + [action_flat(120)] = 1092
    Q1, Q2: 1092 → 512 → ReLU → 512 → ReLU → 1
    """

    def __init__(self, z_rl_dim=960, state_dim=12, chunk_size=10, action_dim=12,
                 hidden_dim=512, num_layers=3):
        x_dim = z_rl_dim + state_dim              # 972
        action_flat_dim = chunk_size * action_dim   # 120
        input_dim = x_dim + action_flat_dim         # 1092

        self.q1 = make_mlp(input_dim, hidden_dim, num_layers, output_dim=1)
        self.q2 = make_mlp(input_dim, hidden_dim, num_layers, output_dim=1)

    def forward(self, z_rl, s_p, action) -> (q1, q2):
        """Returns both Q-values for clipped double-Q"""

    def q1_only(self, z_rl, s_p, action) -> q1:
        """For actor gradient (maximize Q1)"""
```

#### RLTTrainer

TD3+BC training orchestrator:

```python
class RLTTrainer:
    """
    TD3+BC trainer:
    - Twin critics with clipped double-Q (TD3)
    - Target policy smoothing: add clipped noise to target action (TD3 §4.3)
    - Delayed actor updates (every 2 critic steps)
    - BC regularization: L_π = -Q1(x,a) + β * ||a - ã||²
    - Reference action dropout (50%)
    - Target networks with Polyak averaging (τ=0.005)
    - Actor/Critic share SAME state input (symmetric design)
    """

    def __init__(self, actor, critic, device,
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, beta=0.1,
                 target_noise_std=0.2, noise_clip=0.5,
                 chunk_size=10, actor_delay=2, grad_clip=1.0):
        self.gamma_chunk = gamma ** chunk_size  # 0.99^10 ≈ 0.904
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)
        self.actor_opt = Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(critic.parameters(), lr=critic_lr)

    def update_critic(self, batch) -> metrics:
        """
        # --- TD3 Target Policy Smoothing (Section 4.3) ---
        # 1. Get deterministic target action from target actor
        next_a = actor_target(z_rl_next, s_p_next, ref_next)
        # 2. Add clipped Gaussian noise (prevents Q overestimation at narrow peaks)
        noise = (torch.randn_like(next_a) * target_noise_std).clamp(-noise_clip, noise_clip)
        next_a_smooth = (next_a + noise).clamp(-1, 1)  # stay in normalized [-1,1] range
        # 3. Clipped double-Q with smoothed target
        target = chunk_return + γ^C * (1-d) * min(Q1_t(z_next, s_next, next_a_smooth),
                                                   Q2_t(z_next, s_next, next_a_smooth))
        # 4. Partial chunks: done=True → (1-d)=0 → target = chunk_return only
        loss = MSE(Q1, target) + MSE(Q2, target)
        """

    def update_actor(self, batch) -> metrics:
        """
        ref_dropped = dropout(ref)              # 50% → zeros (actor input only)
        a = actor(z_rl, s_p, ref_dropped)       # same state as critic
        q = critic.q1_only(z_rl, s_p, a)        # same state
        loss = -q.mean() + β * MSE(a, ref)      # BC 对齐原始 ã，不是 ref_dropped！

        Key: Dropout 只影响 actor INPUT，BC target 始终是 VLA 原始输出 ã。
        当 actor 看不到 ã 时，它必须靠 z_rl 生成接近 ã 的动作。
        """

    def update(self, batch) -> metrics:
        """One full iteration: critic always, actor delayed, target soft update"""

    def _soft_update_targets(self):
        """Polyak: θ_target = τ*θ + (1-τ)*θ_target"""
```

### 2. `scripts/train_rl_token_stage2.py` (~450 lines)

Main training script:

```python
"""
Usage: python -m scripts.train_rl_token_stage2 --config configs/train_rl_stage2.yaml

Requires: Isaac Sim running, Stage 1 checkpoint, SmolVLA checkpoint
"""

def train(cfg):
    device = torch.device(cfg["device"])  # "cuda"

    # Phase 1: Load frozen components
    normalizer = SimpleNormalizer(cfg["dataset_stats_path"], device)
    vla_hook = VLAPrefixHook(pretrained_path=..., device=device, ...)
    vla_policy = SmolVLAPolicy.from_pretrained(...)  # for reference actions
    stage1 = RLTokenStage1(...)
    stage1.load_state_dict(torch.load(stage1_path))
    stage1.eval(); freeze(stage1)

    # Phase 2: Create trainable components
    actor = RLActor(z_rl_dim=960, state_dim=12, ...)
    critic = TwinCritic(z_rl_dim=960, state_dim=12, ...)
    replay = ReplayBuffer(capacity=100000, device=device, ...)
    trainer = RLTTrainer(actor, critic, device=device, ...)

    # Phase 3: Create env (CPU)
    env = create_isaac_env(cfg)  # CPU

    # Phase 4: Warmup with VLA (fill replay buffer, no RL updates)
    for ep in range(N_warm):
        obs = env.reset()
        while not done:
            z_rl, a_tilde, s_p = process_observation(obs, vla_hook, stage1, normalizer, device)
            action_chunk = a_tilde[:, :C, :]               # use VLA directly
            action_raw = normalizer.denormalize_action(action_chunk)
            rewards = []
            for t in range(C):
                obs, reward, done = env.step(action_raw[t].cpu().numpy())
                rewards.append(reward)
                if done:
                    break                                        # early term on episode end
            n_exec = len(rewards)
            chunk_return = sum(gamma**k * r for k, r in enumerate(rewards))
            # handle partial chunk
            action_stored = action_chunk.clone()
            if n_exec < C:
                action_stored[:, n_exec:] = 0                  # zero-pad phantom steps
            next_z_rl, next_s_p = (zeros, zeros) if done else process_observation(obs, ...)
            replay.add(z_rl, s_p, a_tilde, action_stored, chunk_return,
                       next_z_rl, next_s_p, done)

    # Phase 4.5: BC Pretrain Actor (warm start from VLA behavior)
    #   Goal: initialize actor to mimic VLA before RL begins
    #   Uses warmup data already in replay buffer
    bc_optim = Adam(actor.parameters(), lr=1e-3)
    for epoch in range(bc_pretrain_epochs):
        total_bc_loss = 0
        n_batches = 0
        for _ in range(bc_batches_per_epoch):
            batch = replay.sample(batch_size)
            # actor sees full ref (no dropout during BC pretrain)
            a_pred = actor(batch["z_rl"], batch["s_p"], batch["ref_action"])
            loss = F.mse_loss(a_pred, batch["ref_action"][:, :C, :].flatten(1))
            bc_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            bc_optim.step()
            total_bc_loss += loss.item()
            n_batches += 1
        avg_loss = total_bc_loss / n_batches
        print(f"  BC epoch {epoch}: loss={avg_loss:.6f}")
        if avg_loss < bc_loss_threshold:
            print(f"  BC converged at epoch {epoch}")
            break
    # Sync actor target with BC-pretrained weights
    trainer.actor_target = copy.deepcopy(actor)

    # Phase 5: Online RL
    for ep in range(total_episodes):
        obs = env.reset()
        while not done:
            z_rl, a_tilde, s_p = process_observation(obs, vla_hook, stage1, normalizer, device)
            action_norm = actor(z_rl, s_p, a_tilde)        # ref dropout inside forward()
            action_raw = normalizer.denormalize_action(action_norm)
            for t in range(C):
                obs, reward, done = env.step(action_raw[t].cpu().numpy())
            replay.add(...)
            for g in range(G):
                batch = replay.sample(batch_size)
                metrics = trainer.update(batch)

        # Log & checkpoint
```

Key helper: `process_observation()` — unified VLA hook + z_rl encoding + state assembly:

```python
@torch.no_grad()
def process_observation(obs_dict, vla_hook, stage1, normalizer, device):
    """
    Returns: z_rl(1,960), a_tilde(1,50,12), s_p(1,12)
    
    Uses VLAStage2Hook for a single VLM forward to get z_vlm and ã.
 
    """
    # 1. VLA unified forward — 一次 VLM forward, 同时获取 z_vlm 和 ã
 batch = prepare_vla_batch(obs_dict, device)    z_vlm = vla_hook.extract_prefix(batch)           # (1, 196, 960) on GPU

    # 2. VLA reference actions (full SmolVLA pipeline)
    a_tilde = vla_sample_actions(batch, vla_hook, vla_policy)  # (1, 50, 12) on GPU

    # 3. RL Token encode
    z_target = stage1.apply_keep_mask(z_vlm)          # (1, 193, 960)
    z_rl = stage1.encoder(z_target)                   # (1, 960) — no normalization

    # 4. Proprioceptive state (normalize with dataset stats)
    joint_pos = torch.as_tensor(obs_dict["observation.state"],
                                dtype=torch.float32, device=device).unsqueeze(0)
    s_p = normalizer.normalize_state(joint_pos)       # (1, 12) normalized

    return z_rl, a_tilde, s_p
```

### 3. `configs/train_rl_stage2.yaml`

```yaml
# Stage 1 artifacts
smolvla_pretrained_path: outputs/moe_train/.../pretrained_model
rl_token_stage1_path: outputs/rl_token/stage1/checkpoints/step_10000/rl_token_stage1.pt  # step 10K: best action info (MSE 0.0275, -5.5% vs state), best episode gap (0.74); step 45K overfits reconstruction at expense of task relevance
task_description: "fold the garment"

# Normalization (dataset stats only)
dataset_stats_path: Datasets/example/top_long_merged/meta/stats.json

# Environment (CPU)
garment_name: Top_Long_Unseen_0
garment_type: top_long

# Architecture (symmetric actor/critic)
chunk_size: 10
z_rl_dim: 960
state_dim: 12                 # joint_pos only (same for actor & critic)
action_dim: 12
hidden_dim: 512
num_layers: 3
fixed_std: 0.0067             # exp(-5)
ref_dropout: 0.5

# Training
actor_lr: 3.0e-4
critic_lr: 3.0e-4
gamma: 0.99                 # per-step discount; trainer uses gamma^C = 0.99^10 ≈ 0.904 for chunk-level TD targets
tau: 0.005
beta: 0.1
actor_delay: 2
target_noise_std: 0.2          # TD3 target policy smoothing σ
noise_clip: 0.5                # TD3 noise clip c
grad_clip: 1.0
update_to_data_ratio: 5

# Replay buffer
replay_capacity: 100000
warmup_episodes: 20

# BC pretrain (Phase 4.5: actor warm start)
bc_pretrain_epochs: 100        # max epochs
bc_batches_per_epoch: 100      # batches per epoch
bc_lr: 1.0e-3                  # BC learning rate
bc_loss_threshold: 0.01        # stop early if loss < this

# Run
total_episodes: 500
batch_size: 256
output_dir: outputs/rl_token/stage2
save_freq: 50
device: cuda                # GPU for policy
env_device: cpu             # CPU for garment sim
seed: 42
```

### 4. `scripts/eval_policy/rlt_policy.py` (~200 lines)

```python
@PolicyRegistry.register("rlt")
class RLTPolicy(BasePolicy):
    """
    Pipeline: Every C steps → VLA(z_vlm + ã) → z_rl → Actor → action chunk → execute C steps
    Uses full VLA reference at eval (paper-faithful).
    """

    def __init__(self, smolvla_pretrained_path=None, rl_token_stage1_path=None,
                 dataset_stats_path=None, task_description="fold the garment",
                  chunk_size=10, device="cuda"):
        self.vla_hook = VLAStage2Hook(pretrained_path=smolvla_pretrained_path, device=device)
        self.stage1 = load_frozen_stage1(rl_token_stage1_path, device)
        self.actor = load_trained_actor(actor_path, device)
        self.normalizer = SimpleNormalizer(dataset_stats_path, device)
        self.chunk_size = chunk_size
        self._action_queue = []

    def reset(self):
        self._action_queue = []

    def select_action(self, observation):
        if len(self._action_queue) == 0:
            self._replan(observation)
        return self._action_queue.pop(0)

    def _replan(self, observation):
        with torch.no_grad():
            # 1. VLA unified forward (GPU) — 一次 VLM forward, shared KV cache
            z_vlm, a_tilde = self.vla_hook.forward(observation)
            a_tilde_c = a_tilde[:, :self.chunk_size, :]

            # 2. RL Token (GPU)
            z_target = self.stage1.apply_keep_mask(z_vlm)
            z_rl = self.stage1.encoder(z_target)  # no normalization

            # 3. Actor state (position only, normalized)
            s_p_raw = torch.as_tensor(observation["observation.state"],
                                      dtype=torch.float32, device=self.device).unsqueeze(0)
            s_p = self.normalizer.normalize_state(s_p_raw)

            # 4. Actor (deterministic, full ref, normalized space)
            action_norm = self.actor.get_deterministic_action(z_rl, s_p, a_tilde)
            # 5. Denormalize back to raw joint space for env
            action_raw = self.normalizer.denormalize_action(action_norm)
            actions = action_raw.squeeze(0).cpu().numpy()  # (C, 12)
            self._action_queue = list(actions)
```

---

## Files to Modify

### 5. `garment_bi_v2.py` — **No modifications needed**

All state assembly happens in `process_observation()` using only `observation.state`
from the env obs dict. No privileged info (vel/ee) needed anymore.

### 6. `scripts/eval_policy/__init__.py`

Add: `from .rlt_policy import RLTPolicy`

---

## Symmetric Actor/Critic Design

```
Actor input (972 + 120 = 1092):   [deployable]
  [z_rl(960) + joint_pos_norm(12)] + [ref_action_flat(120)]
  → MLP 512 → 512 → 120
  → action_chunk (10, 12) in normalized space

Critic input (972 + 120 = 1092):  [same as actor]
  [z_rl(960) + joint_pos_norm(12)] + [action_flat(120)]
  → MLP 512 → 512 → 1
  → Q-value (scalar)
```

Design rationale:
- **Symmetric**: Actor & Critic share same state input `[z_rl + joint_pos_norm]`
- **Simpler**: No need for privileged info extraction (vel/ee) from Isaac Lab API
- **Dataset stats only**: All normalization from `stats.json`, no warmup stats collection
- **z_rl as-is**: Stage 1 output already well-conditioned (LayerNorm in transformer)

---

## Performance Estimate (GPU)

```
Per chunk decision:
  VLA prefix pass (GPU):      ~30-80ms
  VLA ODE denoising (GPU):    ~20-50ms
  RL Token encode (GPU):      ~2ms
  Actor forward (GPU):        ~1ms
  CPU→GPU transfer:           ~1ms
  GPU→CPU transfer:           ~1ms
  C=10 sim steps (CPU):       ~10ms
  Total:                      ~65-145ms per chunk

Per episode (~180 chunks):    ~12-26 seconds wall-clock
500 episodes:                 ~1.7-3.6 hours
```

This is very feasible!

---

## Verification Plan

1. **Unit tests** (no Isaac Sim needed):
   - `test_actor_forward`: dummy inputs → output shape (B, 10, 12)
   - `test_critic_forward`: dummy inputs → (q1, q2) shape (B,)
   - `test_replay_buffer`: add/sample roundtrip, verify shapes
   - `test_trainer_update`: dummy batch → valid losses

2. **Integration tests** (need Isaac Sim):
   - Warmup: 2 VLA episodes → replay buffer non-empty
   - Single RL step: collect_chunk + trainer.update → losses finite
   - Checkpoint: save/load cycle preserves behavior

3. **Eval test**:
   - Load actor via RLTPolicy → run 5 episodes → valid actions
   - Compare success rate with pure VLA baseline

---

## Implementation Order

### 0. `source/lehome/lehome/models/vla_stage2_hook.py` (~120 lines) — **NEW**

> **NEW**

统一 VLA Hook： 一次 VLM forward 同时产出 z_vlm 和 ã， 共享 KV Cache，避免重复计算

| 2 | `source/lehome/lehome/models/rl_stage2.py` (~400 lines) | Core: Normalizer, Actor, Critic, Buffer, Trainer |


**No env modification needed** — only `observation.state` from standard obs dict.

**Normalization**: Replay buffer stores normalized joint_pos and actions (via `stats.json`),
raw z_rl (no normalization), and raw rewards. Actor output denormalized before `env.step()`.
BC loss in normalized space, naturally aligned with VLA output.

---

## Architecture Diagrams

### 1. Component Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          STAGE 2: 组件架构                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌───────────────────── FROZEN (from Stage 1) ──────────────────────┐      ║
║  │                                                                   │      ║
║  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐  │      ║
║  │  │  SmolVLA     │   │  RL Token    │   │  VLA Action Expert   │  │      ║
║  │  │  VLM Prefix  │──▶│  Encoder     │   │  (ODE Denoiser)      │  │      ║
║  │  │  (SigLIP+    │   │  (2-layer    │   │                      │  │      ║
║  │  │   Gemma)     │   │   transformer│   │  z_vlm ──▶ ã[0:50]  │  │      ║
║  │  │              │   │   + LayerNorm)│   │  (50 steps × 12D)   │  │      ║
║  │  │ images+lang  │   │              │   │                      │  │      ║
║  │  │ ──▶ z_vlm    │   │ z_target     │   └──────────┬───────────┘  │      ║
║  │  │ (196×960)    │   │ ──▶ z_rl     │              │              │      ║
║  │  └──────────────┘   │ (1×960)      │              │ ã (ref)      │      ║
║  │        │             └──────┬───────┘              │              │      ║
║  │        │ z_vlm            │ z_rl                  │              │      ║
║  └────────│──────────────────│──────────────────────│──────────────┘      ║
║           │                  │                      │                      ║
║           │                  ▼                      ▼                      ║
║  ┌───────────────────── TRAINABLE ────────────────────────────────────┐    ║
║  │                                                                    │    ║
║  │  ┌─────────────────────────────────────────────────────────────┐   │    ║
║  │  │                      SimpleNormalizer                       │   │    ║
║  │  │  dataset stats.json (fixed, no running stats)               │   │    ║
║  │  │  • joint_pos:  raw ──▶ normalized  (x - mean) / std        │   │    ║
║  │  │  • action:     raw ──▶ normalized  (x - mean) / std        │   │    ║
║  │  │  • z_rl:       不归一化 (Stage 1 LayerNorm 已控制)          │   │    ║
║  │  └─────────────────────────────────────────────────────────────┘   │    ║
║  │                                                                    │    ║
║  │  ┌──────────────┐     ┌──────────────────────────────────────┐    │    ║
║  │  │   RLActor    │     │           TwinCritic                 │    │    ║
║  │  │              │     │                                      │    │    ║
║  │  │ state + ref  │     │  state + action                     │    │    ║
║  │  │    │         │     │    │                                 │    │    ║
║  │  │    ▼         │     │    ▼                                 │    │    ║
║  │  │ ┌──────────┐ │     │ ┌──────────┐  ┌──────────┐         │    │    ║
║  │  │ │concat+   │ │     │ │  Q1      │  │  Q2      │         │    │    ║
║  │  │ │MLP       │ │     │ │  MLP     │  │  MLP     │         │    │    ║
║  │  │ │1092→512→ │ │     │ │1092→512→ │  │1092→512→ │         │    │    ║
║  │  │ │512→120   │ │     │ │512→1     │  │512→1     │         │    │    ║
║  │  │ └────┬─────┘ │     │ └────┬─────┘  └────┬─────┘         │    │    ║
║  │  │      │       │     │      │             │               │    │    ║
║  │  │      ▼       │     │      ▼             ▼               │    │    ║
║  │  │  μ(10×12)    │     │   Q1(s,a)     Q2(s,a)             │    │    ║
║  │  │  σ=0.0067    │     │      └─────┬───────┘               │    │    ║
║  │  │  (fixed)     │     │            ▼                       │    │    ║
║  │  │      │       │     │   min(Q1, Q2) ← clipped double-Q  │    │    ║
║  │  │      ▼       │     │                                      │    │    ║
║  │  │ a = μ+σ·ε   │     └──────────────────────────────────────┘    │    ║
║  │  │ (reparam.)  │                                               │    ║
║  │  └──────────────┘                                               │    ║
║  │                                                                │    ║
║  │  ┌──────────────────────────────────────────────────────────┐   │    ║
║  │  │                    ReplayBuffer (GPU)                    │   │    ║
║  │  │  circular, capacity=100K, stores chunk-level transitions │   │    ║
║  │  └──────────────────────────────────────────────────────────┘   │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 2. State Assembly (Symmetric Actor/Critic)

```
                    输入组装：对称 Actor/Critic

     ┌─────────────────────────────────────────────────────┐
     │                                                     │
     │   z_rl ──────────────────────────── 不归一化 ──┐    │
     │   (960D, norm≈28)                              │    │
     │                                                 │    │
     │   joint_pos ── (x-mean)/std ── 归一化 ───────┐ │    │
     │   (12D raw)                                   │ │    │
     │                                                ▼ ▼    │
     │                                          ┌─────────┐ │
     │                                          │ concat  │ │
     │                                          │ (972D)  │ │
     │                                          └────┬────┘ │
     │                                               │      │
     │                    ┌──────────────────────────┤      │
     │                    │                          │      │
     │                    ▼                          ▼      │
     │             ┌───────────┐            ┌───────────┐  │
     │             │  Actor    │            │  Critic   │  │
     │             │           │            │           │  │
     │  ref ã ────▶│ +ref(120D)│   action ─▶│ +act(120D)│  │
     │  (10×12)    │           │   (10×12)  │           │  │
     │  50%dropout │  input    │            │  input    │  │
     │  →zeros     │  1092D    │            │  1092D    │  │
     │             └───────────┘            └───────────┘  │
     └─────────────────────────────────────────────────────┘
```

### 3. Training Pipeline (3 Phases)

```
═══════════════════════════════════════════════════════════════════════
                     STAGE 2 训练流程 (3 Phases)
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│  Phase 4: WARMUP — 用 VLA 填充 ReplayBuffer (不训练 RL)            │
│                                                                     │
│  ┌──────────┐    ┌─────────┐    ┌─────────┐    ┌───────────────┐  │
│  │  env obs │───▶│   VLA   │───▶│ RL Token │───▶│ z_rl + ã + s_p│  │
│  │  (CPU)   │    │ z_vlm   │    │ encoder │    │               │  │
│  │          │    │ + ã     │    │  → z_rl  │    │   action = ã  │  │
│  └──────────┘    └─────────┘    └─────────┘    │  (直接用VLA)  │  │
│       │                                         └───────┬───────┘  │
│       │ env.step(ã)                                     │          │
│       ▼                                                 ▼          │
│  ┌──────────┐                                    ┌──────────────┐  │
│  │ rewards  │                                    │ ReplayBuffer │  │
│  │ r_0..r_9 │──── chunk_return ─────────────────▶│ .add(...)    │  │
│  └──────────┘                                    └──────────────┘  │
│                                                                     │
│  重复 N_warm=20 episodes                                            │
│  目的：给 Critic 一个初始学习信号                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 4.5: BC PRETRAIN — 行为克隆初始化 Actor                      │
│                                                                     │
│  ┌──────────────┐     ┌────────────────────────────────────┐       │
│  │ ReplayBuffer │────▶│ sample batch (256)                  │       │
│  │ (warmup data)│     │ z_rl, s_p, ã(50步), action        │       │
│  └──────────────┘     └──────────────┬─────────────────────┘       │
│                                      │                             │
│                    ┌─────────────────┐│                             │
│                    │ 滑动窗口增强     ││                             │
│                    │ stride=2         ││                             │
│                    │ ã[0:10], ã[2:12] ││                            │
│                    │ ã[4:14], ...     ││                            │
│                    │ → ~21 samples    ▼│                            │
│                    │ per VLA forward  │                             │
│                    └────────┬─────────┘                             │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Actor(z_rl, s_p, ã_chunk) ──▶ a_pred                    │     │
│  │  Loss = MSE(a_pred, ã_chunk)                               │     │
│  │  无 dropout (actor 总是看到 ã)                              │     │
│  │  100 epochs, lr=1e-3, early stop at loss < 0.01           │     │
│  └───────────────────────────────────────────────────────────┘     │
│                             │                                       │
│                             ▼                                       │
│  actor_target = deepcopy(actor)  ← 同步 target 网络                 │
│                                                                     │
│  目的：让 Actor 从 VLA 行为克隆开始，避免 random actor 产生垃圾     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 5: ONLINE RL — TD3+BC 训练 (主循环)                         │
│                                                                     │
│  每个 episode:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  COLLECT LOOP (每个 C=10 步执行一次)                        │   │
│  │                                                             │   │
│  │  env obs ──▶ VLA ──▶ z_vlm ──▶ z_rl ──▶ ã[0:10]          │   │
│  │                                         │           │       │   │
│  │                                         │      ┌────┘       │   │
│  │                                         ▼      ▼            │   │
│  │                               Actor(z_rl, s_p, ã)          │   │
│  │                                  │  50% dropout: ã→zeros   │   │
│  │                                  ▼                          │   │
│  │                            a_norm (10×12)                   │   │
│  │                                  │                          │   │
│  │                                  ▼ denormalize              │   │
│  │                            a_raw = a*std+mean               │   │
│  │                                  │                          │   │
│  │                                  ▼ send to CPU              │   │
│  │  ┌───────────────────────────────────────────────────┐     │   │
│  │  │  for t = 0..9:                                    │     │   │
│  │  │    obs_t, r_t, done_t = env.step(a_raw[t])  # CPU │     │   │
│  │  │    rewards.append(r_t)                            │     │   │
│  │  │    if done_t: break  ← episode 边界提前终止        │     │   │
│  │  │                                                    │     │   │
│  │  │  n_exec = len(rewards)  (可能 < C=10)              │     │   │
│  │  │  chunk_return = Σ γ^t · r_t  (只累加实际执行的)    │     │   │
│  │  └───────────────────────────────────────────────────┘     │   │
│  │                          │                                  │   │
│  │                          ▼                                  │   │
│  │  ┌───────────────────────────────────────────────────┐     │   │
│  │  │  ReplayBuffer.add(                                 │     │   │
│  │  │    z_rl, s_p_norm, ã_full(50步),                  │     │   │
│  │  │    a_stored, chunk_return,                         │     │   │
│  │  │    next_z_rl, next_s_p_norm, done                  │     │   │
│  │  │  )                                                 │     │   │
│  │  │                                                    │     │   │
│  │  │  if n_exec < C:                                    │     │   │
│  │  │    a_stored[:, n_exec:] = 0  # zero-pad phantom    │     │   │
│  │  │  if done:                                          │     │   │
│  │  │    next_z_rl = zeros, next_s_p = zeros             │     │   │
│  │  └───────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                          ▼  G=5 iterations                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  UPDATE LOOP (TD3+BC, 全在 GPU)                             │   │
│  │                                                             │   │
│  │  batch = replay.sample(256)                                 │   │
│  │                                                             │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Step 1: CRITIC UPDATE (每次都执行)                  │   │   │
│  │  │                                                      │   │   │
│  │  │  next_a = actor_target(z_rl', s_p', ã')  ← 无噪声   │   │   │
│  │  │  noise = clip(N(0, σ=0.2), [-0.5, 0.5])  ← TD3      │   │   │
│  │  │  next_a_smooth = (next_a + noise)          ← 平滑    │   │   │
│  │  │                                                      │   │   │
│  │  │  target = r + γ^C·(1-d)·min(Q1'(z',s',a_smooth),    │   │   │
│  │  │                            Q2'(z',s',a_smooth))      │   │   │
│  │  │         ↑ γ^C=0.99^10≈0.904                          │   │   │
│  │  │         ↑ done=True → (1-d)=0 → target=r             │   │   │
│  │  │                                                      │   │   │
│  │  │  L_Q = MSE(Q1, target) + MSE(Q2, target)            │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                                                             │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Step 2: ACTOR UPDATE (每2次critic执行1次)           │   │   │
│  │  │                                                      │   │   │
│  │  │  ref_dropped = dropout(ã, p=0.5)  → 50% 变 zeros    │   │   │
│  │  │  a = actor(z_rl, s_p, ref_dropped) ← 注意用dropped  │   │   │
│  │  │  Q = critic.q1_only(z_rl, s_p, a)                    │   │   │
│  │  │                                                      │   │   │
│  │  │  L_π = -Q.mean() + β·MSE(a, ã_original)             │   │   │
│  │  │         ↑ maximize Q    ↑ BC正则化 (β=0.1)           │   │   │
│  │  │         ↑                ↑ 注意: BC target 是原始 ã   │   │   │
│  │  │         ↑                  不是 ref_dropped!          │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  │                                                             │   │
│  │  ┌─────────────────────────────────────────────────────┐   │   │
│  │  │  Step 3: TARGET SOFT UPDATE                          │   │   │
│  │  │                                                      │   │   │
│  │  │  θ_target ← τ·θ + (1-τ)·θ_target   (τ=0.005)       │   │   │
│  │  │  分别更新 actor_target 和 critic_target               │   │   │
│  │  └─────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. Data Space Flow (Normalization)

```
═══════════════════════════════════════════════════════════════
              数据空间流转图 (Normalization Flow)
═══════════════════════════════════════════════════════════════

                    ENV (原始空间)
                    joint_pos: [-1.73, 1.65]
                    action:    同 joint_pos
                         │
            ┌────────────┤
            │            │
            ▼            ▼
     ┌─────────────┐  ┌─────────────┐
     │ normalize   │  │ normalize   │
     │ (stats.json)│  │ (stats.json)│
     │             │  │             │
     │ jp → s_p   │  │ act → a_norm│
     │ (12D, ~N01) │  │ (12D, ~N01) │
     └──────┬──────┘  └──────┬──────┘
            │                │
            ▼                │
     ┌─────────────┐         │
     │  z_rl (960D)│         │
     │  不归一化    │         │
     │  norm ≈ 28  │         │
     └──────┬──────┘         │
            │                │
            ▼                │
     ┌─────────────┐         │
     │ state =     │         │
     │ [z_rl, s_p] │         │
     │ (972D)      │         │
     └──────┬──────┘         │
            │                │
            ▼                ▼
     ┌──────────────────────────────────────────┐
     │          ACTOR / CRITIC 空间             │
     │                                          │
     │  Actor input:  state(972) + ref(120)     │
     │  Actor output: a_pred(120) ← normalized  │
     │  Critic input: state(972) + act(120)     │
     │  Critic output: Q ∈ R                    │
     │                                          │
     │  BC loss: ||a_pred - ã_norm||²           │
     │           ↑ 都在 normalized 空间!         │
     └──────────────────────┬───────────────────┘
                            │
                            ▼ denormalize
                     a_raw = a_pred * std + mean
                            │
                            ▼
                    ENV (原始空间)
                    env.step(a_raw)
```

### 5. Episode Boundary Handling (Partial Chunks)

```
═════════════════════════════════════════════════════════
         Episode 边界：Partial Chunk 处理
═════════════════════════════════════════════════════════

正常情况 (n_exec = C = 10):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  action: [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉]
  exec:   [ ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ ]
  reward: [r₀, r₁, r₂, r₃, r₄, r₅, r₆, r₇, r₈, r₉]

  chunk_return = r₀ + γr₁ + γ²r₂ + ... + γ⁹r₉
  next_z_rl = encode(obs₉)
  done = False

提前终止 (n_exec = 5, done=True at step 5):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  action: [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈, a₉]
  exec:   [ ✓   ✓   ✓   ✓   ✓   ─   ─   ─   ─   ─ ]
  reward: [r₀, r₁, r₂, r₃, r₄]

  a_stored: [a₀, a₁, a₂, a₃, a₄,  0,  0,  0,  0,  0]
                                        ↑ zero-pad phantom

  chunk_return = r₀ + γr₁ + γ²r₂ + γ³r₃ + γ⁴r₄
  next_z_rl = zeros  ← terminal state
  next_s_p  = zeros
  done = True

  TD target = chunk_return + γ^C · (1-done) · Q_target(...)
            = chunk_return + 0
            = chunk_return  ← phantom actions 无影响
```
