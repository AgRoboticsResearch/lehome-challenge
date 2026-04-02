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
| Actor state | **[z_rl(960) + joint_pos(12)] = 972D** | User decision: actor only gets position |
| Critic state | **[z_rl(960) + joint_pos(12) + joint_vel(12) + ee_pose(12)] = 996D** | User decision: critic gets privileged info (pos+vel+ee_pose) |
| RL algorithm | **TD3+BC** (twin critic, fixed sigma, BC regularization) | Paper Section IV-B |
| Reward | **Dense reward** (existing 0-1 from env) | ~144x richer signal than paper's sparse +1/0 |
| Reference dropout | **50%** during training, always provided at eval | Paper Section IV-B |
| Eval ã | **Full VLA at eval** | Paper-faithful, fast enough on GPU |

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

2. VLA FORWARD on GPU (frozen, ~50-200ms)
   z_vlm = vla_hook.extract_prefix(obs_gpu)              # (1, 196, 960)
   a_tilde = vla.sample_actions(...)[:, :C, :]           # (1, 10, 12)

3. RL TOKEN on GPU (frozen, fast)
   z_target = apply_keep_mask(z_vlm)                     # (1, 193, 960)
   z_rl = rl_token_encoder(z_target)                     # (1, 960)

4. ASSEMBLE STATES
   actor_x = [z_rl(960), joint_pos(12)]                          # (972,)
   critic_x = [z_rl(960), joint_pos(12), joint_vel(12), ee_pose(12)]  # (996,)

5. ACTOR on GPU (trainable, ~1ms)
   50% dropout: ref = ã OR zeros
   action_chunk = actor(actor_x, ref)                    # (1, 10, 12)
   action_cpu = action_chunk.cpu().numpy()

6. EXECUTE open-loop on CPU (C=10 sim steps)
   for t in range(C):
     obs_t, reward_t, done_t = env.step(action_cpu[t])  # CPU sim
   Collect: rewards[], done

7. STORE in replay buffer (on GPU)
   chunk_return = Σ γ^t * r_t
   (z_rl, s_p_actor, s_p_critic, ref, action, chunk_return, next_z_rl, next_s_p_*, done)

8. OFF-POLICY UPDATE on GPU (G=5 iterations)
   batch = replay_buffer.sample(256)
   2x critic_update → 1x actor_update → target soft update
```

---

## Files to Create

### 1. `source/lehome/lehome/models/rl_stage2.py` (~450 lines)

Core Stage 2 components:

#### ReplayBuffer

Circular buffer storing chunk-level transitions on GPU:

```
Fields (each a pre-allocated tensor):
  z_rl:          (capacity, 960)
  s_p_actor:     (capacity, 12)        # joint_pos only
  s_p_critic:    (capacity, 36)        # joint_pos(12) + joint_vel(12) + ee_pose(12)
  ref_action:    (capacity, 10, 12)    # VLA reference (ALWAYS original ã, dropout is in actor.forward())
  action:        (capacity, 10, 12)    # actor output
  reward:        (capacity,)           # discounted chunk return
  next_z_rl:     (capacity, 960)
  next_s_p_critic: (capacity, 36)
  done:          (capacity,)           # bool

Methods:
  add(z_rl, s_p_actor, s_p_critic, ref, action, reward, next_z_rl, next_s_p_critic, done)
  sample(batch_size) → dict of tensors
  __len__() → int
```

#### RLActor

Gaussian policy with asymmetric input:

```python
class RLActor(nn.Module):
    """
    Input:  [z_rl(960) + joint_pos(12)] + [ref_action_flat(120)] = 1092
    MLP:    1092 → 512 → ReLU → 512 → ReLU → 120
    Output: mu(10,12), fixed σ = exp(-5) ≈ 0.0067

    Note: Actor does NOT see velocity. Only position + z_rl + ref.
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
        """Training forward: reparameterization a = mu + σ·ε, no log_prob needed (TD3 not SAC)"""

    def get_deterministic_action(self, z_rl, s_p, ref_action) -> action:
        """Eval forward: return mu without noise"""

    def apply_ref_dropout(self, ref_action, training=True) -> ref_action:
        """50% dropout: replace with zeros"""
```

#### TwinCritic

Ensemble of 2 Q-functions with velocity-augmented state:

```python
class TwinCritic(nn.Module):
    """
    Input:  [z_rl(960) + joint_pos(12) + joint_vel(12) + ee_pose(12)] + [action_flat(120)] = 1116
    Q1, Q2: 1116 → 512 → ReLU → 512 → ReLU → 1
    """

    def __init__(self, z_rl_dim=960, state_dim=36, chunk_size=10, action_dim=12,
                 hidden_dim=512, num_layers=3):
        x_dim = z_rl_dim + state_dim             # 996
        action_flat_dim = chunk_size * action_dim  # 120
        input_dim = x_dim + action_flat_dim        # 1116

        self.q1 = make_mlp(input_dim, hidden_dim, num_layers, output_dim=1)
        self.q2 = make_mlp(input_dim, hidden_dim, num_layers, output_dim=1)

    def forward(self, z_rl, s_p_critic, action) -> (q1, q2):
        """Returns both Q-values for clipped double-Q"""

    def q1_only(self, z_rl, s_p_critic, action) -> q1:
        """For actor gradient (maximize Q1)"""
```

#### RLTTrainer

TD3+BC training orchestrator:

```python
class RLTTrainer:
    """
    TD3+BC trainer:
    - Twin critics with clipped double-Q (TD3)
    - Delayed actor updates (every 2 critic steps)
    - BC regularization: L_π = -Q1(x,a) + β * ||a - ã||²
    - Reference action dropout (50%)
    - Target networks with Polyak averaging (τ=0.005)
    - Actor/Critic have DIFFERENT state inputs (asymmetric)
    """

    def __init__(self, actor, critic, device,
                 actor_lr=3e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, beta=0.1,
                 actor_delay=2, grad_clip=1.0):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = copy.deepcopy(actor)
        self.critic_target = copy.deepcopy(critic)
        # Freeze targets
        self.actor_opt = Adam(actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(critic.parameters(), lr=critic_lr)

    def update_critic(self, batch) -> metrics:
        """
        target = r + γ * (1-d) * min(Q1_t, Q2_t)
        loss = MSE(Q1, target) + MSE(Q2, target)
        """

    def update_actor(self, batch) -> metrics:
        """
        ref_dropped = dropout(ref)              # 50% → zeros (actor input only)
        a = actor(z_rl, s_p_actor, ref_dropped) # actor uses position only
        q = critic.q1_only(z_rl, s_p_critic, a) # critic uses pos+vel
        loss = -q.mean() + β * MSE(a, ref)     # BC 对齐原始 ã，不是 ref_dropped！

        Key: Dropout 只影响 actor INPUT，BC target 始终是 VLA 原始输出 ã。
        当 actor 看不到 ã 时，它必须靠 z_rl 生成接近 ã 的动作。
        """

    def update(self, batch) -> metrics:
        """One full iteration: critic always, actor delayed, target soft update"""

    def _soft_update_targets(self):
        """Polyak: θ_target = τ*θ + (1-τ)*θ_target"""
```

### 2. `scripts/train_rl_token_stage2.py` (~500 lines)

Main training script:

```python
"""
Usage: python -m scripts.train_rl_token_stage2 --config configs/train_rl_stage2.yaml

Requires: Isaac Sim running, Stage 1 checkpoint, SmolVLA checkpoint
"""

def train(cfg):
    device = torch.device(cfg["device"])  # "cuda"

    # Phase 1: Load frozen components
    vla_hook = VLAPrefixHook(pretrained_path=..., device=device, ...)
    vla_policy = SmolVLAPolicy.from_pretrained(...)  # for reference actions
    stage1 = RLTokenStage1(...)
    stage1.load_state_dict(torch.load(stage1_path))
    stage1.eval(); freeze(stage1)

    # Phase 2: Create trainable components
    actor = RLActor(z_rl_dim=960, state_dim=12, ...)
    critic = TwinCritic(z_rl_dim=960, state_dim=36, ...)
    replay = ReplayBuffer(capacity=100000, device=device, ...)
    trainer = RLTTrainer(actor, critic, device=device, ...)

    # Phase 3: Create env (CPU)
    env = create_isaac_env(cfg)  # CPU

    # Phase 4: Warmup with VLA
    for ep in range(N_warm):
        obs = env.reset()
        while not done:
            # VLA produces actions (both z_rl and ã)
            z_rl, a_tilde, s_p_a, s_p_c = process_observation(obs, vla_hook, stage1, env)
            # Use ã directly (no actor perturbation)
            action_chunk = a_tilde[:, :C, :]
            # Execute on CPU
            for t in range(C):
                obs, reward, done = env.step(action_chunk[t].cpu().numpy())
            # Store transition
            next_z_rl, _, next_s_p_c = process_observation(...) if not done else zeros
            replay.add(z_rl, s_p_a, s_p_c, a_tilde, action_chunk, chunk_return,
                       next_z_rl, next_s_p_c, done)

    # Phase 5: Online RL
    for ep in range(total_episodes):
        obs = env.reset()
        while not done:
            # VLA forward (GPU)
            z_rl, a_tilde, s_p_a, s_p_c = process_observation(obs, vla_hook, stage1, env)
            # Actor produces action (GPU)
            action_chunk = actor(z_rl, s_p_a, a_tilde)  # ref dropout inside forward()
            # Execute on CPU
            for t in range(C):
                obs, reward, done = env.step(action_chunk[t].cpu().numpy())
            # Store transition
            replay.add(...)
            # Off-policy updates (GPU)
            for g in range(G):
                batch = replay.sample(batch_size)
                metrics = trainer.update(batch)

        # Log & checkpoint
```

Key helper: `process_observation()` — combines VLA prefix extraction + z_rl encoding + state assembly:

```python
@torch.no_grad()
def process_observation(obs_dict, vla_hook, stage1, env, device):
    """
    Returns: z_rl(1,960), a_tilde(1,50,12), s_p_actor(1,12), s_p_critic(1,36)
    """
    # 1. VLA prefix -> z_vlm
    batch = prepare_vla_batch(obs_dict, device)
    z_vlm = vla_hook.extract_prefix(batch)           # (1, 196, 960) on GPU

    # 2. VLA reference actions (full SmolVLA pipeline)
    a_tilde = vla_sample_actions(batch, vla_hook, vla_policy)  # (1, 50, 12) on GPU

    # 3. RL Token encode
    z_target = stage1.apply_keep_mask(z_vlm)          # (1, 193, 960)
    z_rl = stage1.encoder(z_target)                   # (1, 960)

    # 4. Proprioceptive state
    joint_pos = torch.as_tensor(obs_dict["observation.state"], dtype=torch.float32, device=device)

    # --- Privileged info from Isaac Lab Articulation (direct, no env modification) ---
    # joint_vel: from Articulation.data.joint_vel (verified API)
    left_vel = env.left_arm.data.joint_vel[0].to(device)    # (6,) on GPU
    right_vel = env.right_arm.data.joint_vel[0].to(device)   # (6,)
    joint_vel = torch.cat([left_vel, right_vel])              # (12,)

    # ee_pose: from body_link_pos_w / body_link_quat_w (verified API, last link = gripper)
    from scipy.spatial.transform import Rotation as R
    left_pos = env.left_arm.data.body_link_pos_w[0, -1].to(device)      # (3,)
    left_quat = env.left_arm.data.body_link_quat_w[0, -1].cpu().numpy()  # (4,) xyzw
    left_euler = torch.tensor(R.from_quat(left_quat).as_euler('xyz'),
                              dtype=torch.float32, device=device)          # (3,)
    right_pos = env.right_arm.data.body_link_pos_w[0, -1].to(device)
    right_quat = env.right_arm.data.body_link_quat_w[0, -1].cpu().numpy()
    right_euler = torch.tensor(R.from_quat(right_quat).as_euler('xyz'),
                               dtype=torch.float32, device=device)
    ee_pose = torch.cat([left_pos, left_euler, right_pos, right_euler])  # (12,)

    s_p_actor = joint_pos.unsqueeze(0)                                        # (1, 12)
    s_p_critic = torch.cat([joint_pos, joint_vel, ee_pose]).unsqueeze(0)      # (1, 36)

    return z_rl, a_tilde, s_p_actor, s_p_critic
```

### 3. `configs/train_rl_stage2.yaml`

```yaml
# Stage 1 artifacts
smolvla_pretrained_path: outputs/moe_train/.../pretrained_model
rl_token_stage1_path: outputs/rl_token/stage1/checkpoints/best/rl_token_stage1.pt
task_description: "fold the garment"

# Environment (CPU)
garment_name: Top_Long_Unseen_0
garment_type: top_long

# Architecture
chunk_size: 10
z_rl_dim: 960
actor_state_dim: 12          # joint_pos only
critic_state_dim: 36         # joint_pos(12) + joint_vel(12) + ee_pose(12)
action_dim: 12
actor_hidden_dim: 512
actor_num_layers: 3
critic_hidden_dim: 512
critic_num_layers: 3
fixed_std: 0.0067            # exp(-5)
ref_dropout: 0.5

# Training
actor_lr: 3.0e-4
critic_lr: 3.0e-4
gamma: 0.99
tau: 0.005
beta: 0.1
actor_delay: 2
grad_clip: 1.0
update_to_data_ratio: 5

# Replay buffer
replay_capacity: 100000
warmup_episodes: 20

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

    def __init__(self, smolvla_pretrained_path, rl_token_stage1_path, actor_path,
                 task_description="fold the garment", chunk_size=10, device="cuda"):
        self.vla_hook = VLAPrefixHook(pretrained_path=..., device=device)
        self.vla_policy = SmolVLAPolicy.from_pretrained(...)  # for ã at eval
        self.stage1 = load_frozen_stage1(rl_token_stage1_path, device)
        self.actor = load_trained_actor(actor_path, device)
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
            # 1. VLA forward (GPU) — get z_vlm AND ã
            batch = prepare_vla_batch(observation, self.device)
            z_vlm = self.vla_hook.extract_prefix(batch)
            a_tilde = vla_sample_actions(batch, self.vla_hook, self.vla_policy)[:, :C, :]

            # 2. RL Token (GPU)
            z_target = self.stage1.apply_keep_mask(z_vlm)
            z_rl = self.stage1.encoder(z_target)

            # 3. Actor state (position only)
            s_p = torch.as_tensor(observation["observation.state"],
                                  dtype=torch.float32, device=self.device).unsqueeze(0)

            # 4. Actor (deterministic, full ref)
            action_chunk = self.actor.get_deterministic_action(z_rl, s_p, a_tilde)
            actions = action_chunk.squeeze(0).cpu().numpy()  # (C, 12)
            self._action_queue = list(actions)
```

---

## Files to Modify

### 5. `garment_bi_v2.py` — **No modifications needed**

**Design choice (Option B)**: ee_pose and joint_vel are extracted directly from
`env.left_arm.data` / `env.right_arm.data` in the training script's
`process_observation()` helper. This keeps env modifications to zero and avoids
adding scipy dependency to the sim code.

All state assembly happens in `process_observation()` (see File 2 above).

### 6. `scripts/eval_policy/__init__.py`

Add: `from .rlt_policy import RLTPolicy`

---

## Asymmetric Actor/Critic Design

```
Actor input (972 + 120 = 1092):   [deployable]
  [z_rl(960) + joint_pos(12)] + [ref_action_flat(120)]
  → MLP 512 → 512 → 120
  → action_chunk (10, 12)

Critic input (996 + 120 = 1116):  [privileged, training-only]
  [z_rl(960) + joint_pos(12) + joint_vel(12) + ee_pose(12)] + [action_flat(120)]
  → MLP 512 → 512 → 1
  → Q-value (scalar)
```

Asymmetric state breakdown:
- **Actor** sees only deployable info: joint_pos (from encoders) + z_rl (from camera) + ã (from VLA). No velocity, no ee_pose — nothing that requires FK or velocity sensors.
- **Critic** sees privileged training-only info: everything the actor sees PLUS joint_vel (richer dynamics info) and ee_pose(12) = 2 × (pos_xyz + euler_xyz) — gives critic explicit task-space awareness for better value estimation.
- **ee_pose extraction**: via forward kinematics from `env.left_arm.data.ee_pos`, `env.right_arm.data.ee_pos` (Isaac Lab Articulation provides this).

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

| Step | File | Lines | Notes |
|------|------|-------|-------|
| 1 | `configs/train_rl_stage2.yaml` | ~50 | Config first |
| 2 | `source/lehome/lehome/models/rl_stage2.py` | ~450 | Core: Actor, Critic, Buffer, Trainer |
| 3 | `scripts/train_rl_token_stage2.py` | ~500 | Training loop (reads ee_pose/vel directly from env) |
| 4 | `scripts/eval_policy/rlt_policy.py` | ~200 | Eval wrapper |
| 5 | `scripts/eval_policy/__init__.py` | ~1 | Register policy |

**No env modification needed** — privileged info (joint_vel, ee_pose) extracted
directly from Isaac Lab Articulation API in training/eval scripts.
