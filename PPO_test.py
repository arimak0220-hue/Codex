"""
PPO for MiniWorld CollectHealth-v0  —  RBE-595 Final Project
=============================================================

Architecture
------------
* Shared CNN backbone (3 strided conv layers -> 256-unit MLP) with separate
  actor (policy) and critic (value) heads. Sharing the CNN backbone halves CNN
  parameters and dramatically improves sample efficiency because gradients from
  both heads update the same visual features.
* Frame stacking (FRAME_STACK=4): short-term temporal context via channel concat.
* GAE advantage estimation (O(n) reverse scan), advantage normalisation,
  clipped surrogate loss, value-function loss clipping, entropy bonus with
  linear decay, gradient clipping (max-norm 0.5).

Windows OpenGL fix
------------------
MiniWorld on Windows crashes with exit code 0xC0000374 (heap corruption)
whenever a new OpenGL context is created after the previous one is torn down.

Rules enforced here:
  1. Exactly ONE gym.make() call per process.
  2. The training env is NEVER closed inside train() — it is returned and
     passed directly into evaluate() and closed only once, at the very end
     of __main__.
  3. For --eval-only, the eval env is created ONCE before Agent is built and
     is passed into evaluate(); no probe env is needed because n_actions is
     read from the same env object.
  4. render_mode is ALWAYS "rgb_array" so env.render() works without a second
     GL context. Videos are written with imageio — no RecordVideo / moviepy.

PPO improvements over baseline
-------------------------------
  * Shared CNN backbone   — halves CNN parameters; joint gradient signal
  * Orthogonal init       — all layers; prevents gradient imbalance at init
  * Deeper MLP            — 3456->512->512 avoids single-step bottleneck
  * LayerNorm after CNN   — stabilises feature scale across rollout batches
  * O(n) GAE              — reverse-scan replaces O(n²) nested loop
  * Returns bug fixed     — critic now learns from raw (un-normalised) returns
  * Reward normalisation  — running mean/std keeps gradient magnitudes stable
  * Advantage normalise   — zero-mean / unit-var before each update epoch
  * Value clipping        — clips V-loss like actor ratio (standard PPO)
  * Entropy decay         — linear schedule from ENT_START=0.05 to ENT_END=0.002
  * Larger rollout T=2048 — lower-variance gradient estimates (was T=512)

Outputs
-------
* checkpoint/                 — shared backbone + actor/critic heads (.pt)
* videos/training/            — mp4 every VIDEO_EVERY episodes
* videos/eval/                — mp4 for every evaluation episode
* plots/obs_and_actions.png   — observation + action space (task 1)
* plots/reward_vs_timesteps.png
* plots/survival_vs_timesteps.png
* plots/eval_scores.png
* results/summary.txt

Install
-------
    pip install miniworld gymnasium torch torchvision imageio[ffmpeg] matplotlib

Run
---
    python PPO_test_vid.py                            # train then eval
    python PPO_test_vid.py --lr 1e-4 --n-episodes 3000
    python PPO_test_vid.py --eval-only                # load checkpoint & eval
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions.categorical import Categorical

import gymnasium as gym
import miniworld  # noqa: F401  — registers MiniWorld envs
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Running statistics for reward normalisation
# ===========================================================================
class RunningMeanStd:
    """
    Tracks running mean and variance of a scalar stream using Welford's
    online algorithm. Used to normalise rewards to zero-mean / unit-variance,
    which stabilises training when raw rewards have small or inconsistent scale.

    Without normalisation, rewards of +0.2/step produce very small gradient
    signals that are easily swamped by noise, especially early in training.
    """
    def __init__(self, eps=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps   # small init avoids div-by-zero on first update

    def update(self, x):
        batch_mean = float(np.mean(x))
        batch_var  = float(np.var(x))
        batch_n    = len(x) if hasattr(x, '__len__') else 1

        tot = self.count + batch_n
        new_mean = (self.count * self.mean + batch_n * batch_mean) / tot
        # parallel variance formula
        m_a = self.var   * self.count
        m_b = batch_var  * batch_n
        m_2 = m_a + m_b + (self.mean - batch_mean) ** 2 * self.count * batch_n / tot
        self.var   = m_2 / tot
        self.mean  = new_mean
        self.count = tot

    def normalise(self, x, clip=10.0):
        """Subtract running mean, divide by running std, clip to [-clip, clip]."""
        normed = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -clip, clip)


# ===========================================================================
# Hyperparameters
# ===========================================================================
ENV_ID        = "MiniWorld-CollectHealth-v0"
IMG_H, IMG_W  = 60, 80
FRAME_STACK   = 4

T             = 2560     # rollout length — larger = lower-variance gradient estimates.
                         # 512 was too small (~10 episodes/update, signal too noisy).
BATCH_SIZE    = 64
N_EPOCHS      = 4
LR            = 1e-4     # lowered from 3e-4 — more stable for visual tasks
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
POLICY_CLIP   = 0.2
VALUE_CLIP    = 0.2
MAX_GRAD_NORM = 0.5

ENT_START     = 0.05     # raised from 0.02 — 3D maze needs real exploration early
ENT_END       = 0.002    # entropy coefficient at final episode

N_EPISODES    = 50000
EVAL_EPISODES = 10
VIDEO_EVERY   = 1000
VIDEO_FPS     = 15       # 15 fps — smooth enough and keeps file sizes small
VIDEO_MIN_STEPS = 600    # minimum env steps per video clip (600 / 15fps = 40 sec)
                         # If an episode ends before this, the env is reset and
                         # recording continues — so the video always shows the
                         # agent exploring across multiple episodes.

CHKPT_DIR     = "checkpoint"
PLOT_DIR      = "plots"
RESULTS_DIR   = "results"
TRAIN_VID_DIR = "videos/training"
EVAL_VID_DIR  = "videos/eval"

ACTION_LABELS = [
    "Turn Left", "Turn Right", "Move Forward",
    "Move Back", "Strafe Left", "Strafe Right", "Pick Up",
]


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="PPO — MiniWorld-CollectHealth-v0 (RBE-595)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval-only",     action="store_true")
    p.add_argument("--lr",            type=float, default=LR)
    p.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--n-epochs",      type=int,   default=N_EPOCHS)
    p.add_argument("--n-episodes",    type=int,   default=N_EPISODES)
    p.add_argument("--gamma",         type=float, default=GAMMA)
    p.add_argument("--policy-clip",   type=float, default=POLICY_CLIP)
    p.add_argument("--frame-stack",   type=int,   default=FRAME_STACK)
    p.add_argument("--eval-episodes", type=int,   default=EVAL_EPISODES)
    p.add_argument("--video-every",   type=int,   default=VIDEO_EVERY,
                   help="Record training video every N episodes. 0 = never.")
    return p.parse_args()


# ===========================================================================
# Frame-stack wrapper
# ===========================================================================
class FrameStack:
    """Concatenates N most recent (H,W,3) frames along channel axis."""
    def __init__(self, n, obs_shape):
        self.n = n
        h, w, c = obs_shape
        self.frames = deque([np.zeros((h, w, c), dtype=np.uint8)] * n, maxlen=n)

    def reset(self, obs):
        for _ in range(self.n):
            self.frames.append(obs)
        return self._stack()

    def step(self, obs):
        self.frames.append(obs)
        return self._stack()

    def _stack(self):
        return np.concatenate(list(self.frames), axis=-1)  # (H, W, C*N)


# ===========================================================================
# Observation pre-processing
# ===========================================================================
def preprocess(obs):
    """(H,W,C) uint8  ->  (1,C,H,W) float32 in [0,1]."""
    t = torch.from_numpy(obs.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0)


# ===========================================================================
# PPO Memory
# ===========================================================================
class PPOMemory:
    def __init__(self, batch_size):
        self.states   = []
        self.logprobs = []
        self.vals     = []
        self.actions  = []
        self.rewards  = []
        self.dones    = []
        self.batch_size = batch_size

    def store(self, state, action, logprob, val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def generate_batches(self):
        n       = len(self.states)
        idx     = np.arange(n, dtype=np.int64)
        np.random.shuffle(idx)
        batches = [idx[i: i + self.batch_size] for i in range(0, n, self.batch_size)]
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.logprobs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def clear(self):
        self.states   = []
        self.logprobs = []
        self.actions  = []
        self.rewards  = []
        self.dones    = []
        self.vals     = []


# ===========================================================================
# Weight initialisation
# ===========================================================================
def init_orthogonal(layer, gain=np.sqrt(2), bias=0.0):
    """
    Orthogonal weight initialisation with given gain.
    Standard for PPO — produces well-conditioned weight matrices at init so
    gradients flow evenly through the network from the first update.
    Random (default) init causes gradient imbalance that can prevent learning
    entirely for hundreds of episodes.
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


# ===========================================================================
# Shared CNN + Actor/Critic heads
# ===========================================================================
class CNNBackbone(nn.Module):
    """
    3 strided conv layers: (C, 60, 80) -> 512-dim feature vector.

    Changes vs previous version:
    - All conv layers initialised orthogonally (gain=sqrt(2))
    - Added LayerNorm after flattening — stabilises the distribution of
      features fed into the actor/critic heads across the whole rollout
    - Intermediate MLP layer (3456 -> 512) instead of direct (3456 -> 256):
      the old single-step compression lost too much spatial information and
      created gradient bottlenecks; two smaller steps preserve more structure.
    """
    def __init__(self, in_channels, feat_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            init_orthogonal(nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)), nn.ReLU(),
            init_orthogonal(nn.Conv2d(32, 64, kernel_size=3, stride=2)),          nn.ReLU(),
            init_orthogonal(nn.Conv2d(64, 64, kernel_size=3, stride=2)),          nn.ReLU(),
            nn.Flatten(),
        )
        # Probe flat dim at construction time
        with torch.no_grad():
            dummy    = torch.zeros(1, in_channels, IMG_H, IMG_W)
            flat_dim = self.conv(dummy).shape[1]

        # Two-step MLP: flat_dim -> 512 -> feat_dim avoids the 3456->256
        # single-step information bottleneck that broke gradient flow.
        self.mlp = nn.Sequential(
            init_orthogonal(nn.Linear(flat_dim, 512)), nn.ReLU(),
            nn.LayerNorm(512),                          # stabilises feature scale
            init_orthogonal(nn.Linear(512, feat_dim)), nn.ReLU(),
        )
        self.feat_dim = feat_dim

    def forward(self, x):
        return self.mlp(self.conv(x))


class ActorCriticNetwork(nn.Module):
    """
    Shared CNN backbone with separate actor (policy) and critic (value) heads.
    Both heads use orthogonal init with small gain on the output layer to keep
    initial action probabilities near-uniform and initial values near zero.
    """
    def __init__(self, n_actions, in_channels, lr, chkpt_dir=CHKPT_DIR):
        super().__init__()
        os.makedirs(chkpt_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_critic_ppo.pt")

        self.backbone = CNNBackbone(in_channels)
        fd = self.backbone.feat_dim

        # Gain=0.01 on output layers keeps initial logits/values near zero,
        # so the policy starts close to uniform and the critic starts near 0.
        self.actor_head = nn.Sequential(
            init_orthogonal(nn.Linear(fd, 256), gain=np.sqrt(2)), nn.ReLU(),
            init_orthogonal(nn.Linear(256, n_actions), gain=0.01),
        )
        self.critic_head = nn.Sequential(
            init_orthogonal(nn.Linear(fd, 256), gain=np.sqrt(2)), nn.ReLU(),
            init_orthogonal(nn.Linear(256, 1), gain=1.0),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        feats  = self.backbone(x)
        logits = self.actor_head(feats)
        value  = self.critic_head(feats)
        return Categorical(logits=logits), value.squeeze(-1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(
            torch.load(self.checkpoint_file, map_location=self.device,
                       weights_only=True)
        )


# ===========================================================================
# Agent
# ===========================================================================
class Agent:
    def __init__(self, n_actions, in_channels,
                 gamma=GAMMA, lr=LR, gae_lambda=GAE_LAMBDA,
                 policy_clip=POLICY_CLIP, value_clip=VALUE_CLIP,
                 batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
                 ent_coef=ENT_START):
        self.gamma       = gamma
        self.policy_clip = policy_clip
        self.value_clip  = value_clip
        self.n_epochs    = n_epochs
        self.gae_lambda  = gae_lambda
        self.ent_coef    = ent_coef

        self.net         = ActorCriticNetwork(n_actions, in_channels, lr)
        self.memory      = PPOMemory(batch_size)
        self.reward_rms  = RunningMeanStd()   # running stats for reward normalisation

    @property
    def device(self):
        return self.net.device

    def _to_tensor(self, obs):
        return preprocess(obs).to(self.device)

    def remember(self, state, action, logprob, val, reward, done):
        self.memory.store(state, action, logprob, val, reward, done)

    def save_models(self):
        print("  [ckpt] saving best model...")
        self.net.save_checkpoint()

    def load_models(self):
        print("  [ckpt] loading model...")
        self.net.load_checkpoint()

    def choose_action(self, obs, greedy=False):
        state_t = self._to_tensor(obs)
        with torch.no_grad():
            dist, value = self.net(state_t)
            action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    # -----------------------------------------------------------------------
    def learn(self):
        (state_arr, action_arr, old_logprob_arr,
         vals_arr, reward_arr, dones_arr, batches) = self.memory.generate_batches()

        # ---- Reward normalisation -----------------------------------------
        # Update running statistics with this rollout's rewards, then
        # normalise. This keeps gradient magnitudes consistent regardless of
        # the raw reward scale, which is critical when rewards are small
        # (+0.2/step) and the agent is dying quickly early in training.
        self.reward_rms.update(reward_arr)
        reward_arr = self.reward_rms.normalise(reward_arr)
        # ------------------------------------------------------------------

        # ---- O(n) GAE reverse scan ----------------------------------------
        n         = len(reward_arr)
        advantage = np.zeros(n, dtype=np.float32)
        gae       = 0.0
        for t in reversed(range(n - 1)):
            next_val     = vals_arr[t + 1] * (1 - int(dones_arr[t]))
            delta        = reward_arr[t] + self.gamma * next_val - vals_arr[t]
            gae          = delta + self.gamma * self.gae_lambda * (1 - int(dones_arr[t])) * gae
            advantage[t] = gae
        # ------------------------------------------------------------------

        # Compute returns from RAW advantages BEFORE normalising them.
        # BUG FIX: the old code did `returns = adv_batch + old_values_t[batch]`
        # where adv_batch used NORMALISED advantages. This made the critic learn
        # to predict a shifted/scaled target that changed every epoch, completely
        # breaking value function learning and therefore advantage estimation.
        returns_arr  = advantage + vals_arr          # (T,)  — raw, unnormalised

        advantage_t  = torch.tensor(advantage,    dtype=torch.float32).to(self.device)
        returns_t    = torch.tensor(returns_arr,  dtype=torch.float32).to(self.device)
        old_values_t = torch.tensor(vals_arr,     dtype=torch.float32).to(self.device)

        for _ in range(self.n_epochs):
            # Normalise advantages per epoch (zero-mean, unit-var) for the
            # actor loss only. Returns used in critic loss are NOT normalised.
            adv_norm = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

            for batch in batches:
                states_t = torch.cat(
                    [preprocess(state_arr[i]) for i in batch], dim=0
                ).to(self.device)

                old_logprobs_t = torch.tensor(
                    old_logprob_arr[batch], dtype=torch.float32
                ).to(self.device)
                actions_t = torch.tensor(
                    action_arr[batch], dtype=torch.long
                ).to(self.device)

                dist, new_values = self.net(states_t)

                entropy      = dist.entropy().mean()
                new_logprobs = dist.log_prob(actions_t)
                log_ratio    = new_logprobs - old_logprobs_t
                prob_ratio   = log_ratio.exp()

                adv_batch = adv_norm[batch]

                # Actor loss (clipped surrogate)
                actor_loss = -torch.min(
                    prob_ratio * adv_batch,
                    torch.clamp(prob_ratio, 1 - self.policy_clip,
                                1 + self.policy_clip) * adv_batch,
                ).mean()

                # Critic loss with value clipping.
                # returns_t[batch] are the true bootstrapped targets — NOT
                # normalised — so the critic learns to predict real values.
                ret_batch = returns_t[batch]
                v_clipped = old_values_t[batch] + torch.clamp(
                    new_values - old_values_t[batch],
                    -self.value_clip, self.value_clip,
                )
                critic_loss = 0.5 * torch.max(
                    (new_values - ret_batch) ** 2,
                    (v_clipped  - ret_batch) ** 2,
                ).mean()

                total_loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy

                self.net.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
                self.net.optimizer.step()

        self.memory.clear()


# ===========================================================================
# Visualisation helpers
# ===========================================================================
def save_obs_and_actions(obs, n_actions, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = (ACTION_LABELS[:n_actions]
              + [f"Action {i}" for i in range(len(ACTION_LABELS), n_actions)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].imshow(obs)
    axes[0].set_title("Agent First-Person Observation\n(CollectHealth-v0  60×80 RGB)", fontsize=11)
    axes[0].axis("off")

    colors = plt.cm.Set2(np.linspace(0, 1, n_actions))
    axes[1].bar(range(n_actions), [1] * n_actions, color=colors, edgecolor="white")
    axes[1].set_xticks(range(n_actions))
    axes[1].set_xticklabels(labels[:n_actions], rotation=35, ha="right", fontsize=9)
    axes[1].set_yticks([])
    axes[1].set_title(f"Discrete Action Space  ({n_actions} actions)", fontsize=11)
    axes[1].set_xlabel("Action index", fontsize=10)
    for i in range(n_actions):
        axes[1].text(i, 0.5, str(i), ha="center", va="center",
                     fontsize=11, fontweight="bold", color="white")

    fig.suptitle("RBE-595 — PPO CollectHealth | Observations & Actions",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [vis]  obs + actions -> {path}")


def save_reward_plot(score_history, timestep_history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scores  = np.array(score_history)
    steps   = np.array(timestep_history)
    rolling = np.array([scores[max(0, i - 99): i + 1].mean() for i in range(len(scores))])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, scores,  alpha=0.2, color="steelblue", label="Episode reward")
    ax.plot(steps, rolling, color="steelblue", linewidth=2, label="Rolling mean (100 ep)")
    ax.set_xlabel("Cumulative environment timesteps", fontsize=12)
    ax.set_ylabel("Episode reward", fontsize=12)
    ax.set_title("PPO — CollectHealth-v0 | Reward vs Timesteps",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] reward curve -> {path}")


def save_survival_plot(survival_history, timestep_history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    survival = np.array(survival_history, dtype=np.float32)
    steps    = np.array(timestep_history)
    rolling  = np.array([survival[max(0, i - 99): i + 1].mean() for i in range(len(survival))])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, survival, alpha=0.2, color="mediumseagreen", label="Episode survival steps")
    ax.plot(steps, rolling,  color="mediumseagreen", linewidth=2, label="Rolling mean (100 ep)")
    ax.set_xlabel("Cumulative environment timesteps", fontsize=12)
    ax.set_ylabel("Steps survived per episode", fontsize=12)
    ax.set_title("PPO — CollectHealth-v0 | Survival Duration vs Timesteps",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] survival curve -> {path}")


def save_eval_plot(eval_scores, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mean = float(np.mean(eval_scores))
    std  = float(np.std(eval_scores))
    eps  = list(range(1, len(eval_scores) + 1))

    fig, ax = plt.subplots(figsize=(max(7, len(eval_scores) * 0.9), 5))
    bars = ax.bar(eps, eval_scores, color="coral", edgecolor="white", zorder=3)
    ax.axhline(mean, color="darkred", linewidth=2, linestyle="--",
               label=f"Mean = {mean:.2f}")
    ax.fill_between([0.5, len(eps) + 0.5], mean - std, mean + std,
                    color="darkred", alpha=0.15, label=f"+/-1 Std = {std:.2f}")
    for bar, score in zip(bars, eval_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.5, abs(score) * 0.01),
                f"{score:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Evaluation episode", fontsize=12)
    ax.set_ylabel("Episode score", fontsize=12)
    ax.set_xticks(eps)
    ax.set_title("PPO Evaluation — CollectHealth-v0 | Greedy Policy Scores",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] eval chart -> {path}")


def save_summary(eval_scores, score_history, survival_history, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last_s = score_history[-100:]
    last_v = survival_history[-100:]
    rolling_best = float(np.max([
        np.mean(score_history[max(0, i - 99): i + 1])
        for i in range(len(score_history))
    ]))
    lines = [
        "=" * 62,
        "  RBE-595 Final Project — PPO CollectHealth-v0 Results",
        "=" * 62, "",
        "  Architecture", "  ------------",
        "  Shared CNN backbone + actor/critic heads",
        "  O(n) GAE | Advantage normalisation | Value clipping",
        "  Entropy decay schedule", "",
        "  Hyperparameters", "  ---------------",
        f"  Learning rate      : {cfg['lr']}",
        f"  Batch size         : {cfg['batch_size']}",
        f"  PPO update epochs  : {cfg['n_epochs']}",
        f"  Discount (gamma)   : {cfg['gamma']}",
        f"  GAE lambda         : {GAE_LAMBDA}",
        f"  Policy clip (eps)  : {cfg['policy_clip']}",
        f"  Value clip         : {VALUE_CLIP}",
        f"  Entropy (start/end): {ENT_START} / {ENT_END}",
        f"  Frame stack        : {cfg['frame_stack']}",
        f"  Grad clip max-norm : {MAX_GRAD_NORM}",
        f"  Rollout length (T) : {T}",
        f"  Total episodes     : {cfg['n_episodes']}", "",
        "  Training Results (last 100 episodes)",
        "  -------------------------------------",
        f"  Mean reward        : {float(np.mean(last_s)):.2f}",
        f"  Std reward         : {float(np.std(last_s)):.2f}",
        f"  Mean survival      : {float(np.mean(last_v)):.1f} steps",
        f"  Best rolling avg   : {rolling_best:.2f}", "",
        "  Evaluation Results (greedy/deterministic policy)",
        "  -------------------------------------------------",
        f"  Episodes           : {len(eval_scores)}",
        f"  Mean score         : {float(np.mean(eval_scores)):.2f}",
        f"  Std score          : {float(np.std(eval_scores)):.2f}",
        f"  Best episode       : {float(np.max(eval_scores)):.2f}",
        f"  Worst episode      : {float(np.min(eval_scores)):.2f}", "",
        "  Output Paths", "  ------------",
        f"  Checkpoints        : {CHKPT_DIR}/",
        f"  Training videos    : {TRAIN_VID_DIR}/",
        f"  Eval videos        : {EVAL_VID_DIR}/",
        f"  Plots              : {PLOT_DIR}/",
        "=" * 62,
    ]
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(f"\n  [results] summary -> {path}\n\n" + text)


# ===========================================================================
# Video clip recorder
# ===========================================================================
def record_clip(agent, env, frame_stack, min_steps, path, greedy=True):
    """
    Record a video clip of exactly `min_steps` environment steps, automatically
    resetting the env whenever an episode ends — so the video is always full
    length even if the agent dies in 30 steps.

    Why this fixes the 1-second video problem:
      Early in training the agent survives only ~30-50 steps per episode.
      At 15 fps that is under 4 seconds. By continuing across resets we get
      a full 40-second clip that shows the agent exploring across many episodes,
      making it easy to see what the policy has learned (or not yet learned).

    Parameters
    ----------
    agent       : Agent — policy to record
    env         : open gymnasium env (render_mode must be 'rgb_array')
    frame_stack : int — number of frames to stack
    min_steps   : int — total env steps to capture (resets count as 0 steps)
    path        : str — output .mp4 path
    greedy      : bool — True = deterministic (argmax), False = stochastic
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    raw_obs, _ = env.reset()
    stacker    = FrameStack(frame_stack, raw_obs.shape)
    obs        = stacker.reset(raw_obs)

    frames     = []
    ep_count   = 0
    step       = 0

    # Capture the very first frame before any action
    f = env.render()
    if f is not None:
        frames.append(f)

    while step < min_steps:
        action, _, _ = agent.choose_action(obs, greedy=greedy)
        raw_obs_, reward, terminated, truncated, _ = env.step(action)
        step += 1

        f = env.render()
        if f is not None:
            frames.append(f)

        if terminated or truncated:
            # Episode ended — reset and keep recording without closing the env
            ep_count  += 1
            raw_obs_, _ = env.reset()
            obs          = stacker.reset(raw_obs_)
        else:
            obs = stacker.step(raw_obs_)

    duration_s = len(frames) / VIDEO_FPS
    imageio.mimsave(path, frames, fps=VIDEO_FPS)
    print(f"  [video] {len(frames)} frames | {duration_s:.1f}s | "
          f"{ep_count+1} episode(s) -> {path}")


# ===========================================================================
# Training  (returns env — do NOT close it here; main closes it after eval)
# ===========================================================================
def train(cfg):
    """
    Creates ONE env for the entire training session and RETURNS it without
    closing. The caller (main) passes it to evaluate() and closes it at the
    very end. This is the Windows OpenGL fix: only one env context ever exists.
    """
    for d in [CHKPT_DIR, PLOT_DIR, RESULTS_DIR, TRAIN_VID_DIR, EVAL_VID_DIR]:
        os.makedirs(d, exist_ok=True)

    print("  [env] creating training environment...")
    # render_mode='rgb_array' lets env.render() work without a display window
    env = gym.make(ENV_ID, render_mode="rgb_array")

    n_actions   = env.action_space.n
    obs_shape   = env.observation_space.shape    # (H, W, 3)
    in_channels = 3 * cfg["frame_stack"]

    agent = Agent(
        n_actions   = n_actions,
        in_channels = in_channels,
        gamma       = cfg["gamma"],
        lr          = cfg["lr"],
        policy_clip = cfg["policy_clip"],
        batch_size  = cfg["batch_size"],
        n_epochs    = cfg["n_epochs"],
    )

    print("\n" + "=" * 62)
    print("  RBE-595 — PPO Agent | MiniWorld-CollectHealth-v0")
    print("=" * 62)
    print(f"  Observation space  : {obs_shape}  (H x W x C, first-person RGB)")
    print(f"  Action space       : Discrete({n_actions})")
    print(f"  Frame stack        : {cfg['frame_stack']}  ->  in_channels = {in_channels}")
    print(f"  Device             : {agent.device}")
    print(f"  Shared backbone    : YES  (CNN -> 512-dim -> actor/critic heads)")
    print(f"  Orthogonal init    : YES")
    print()
    print("  Reward structure:")
    print("    +0.2  per timestep alive")
    print("    +1.0  per health kit collected")
    print("    Episode ends when health -> 0")
    print()
    print("  Hyperparameters:")
    for k, v in cfg.items():
        print(f"    {k:<20}: {v}")
    print("=" * 62 + "\n")

    best_score       = float("-inf")
    score_history    = []
    survival_history = []
    timestep_history = []
    learn_iters      = 0
    n_steps          = 0
    obs_vis_saved    = False
    n_episodes       = cfg["n_episodes"]

    for episode in range(n_episodes):
        # Linear entropy decay: high entropy early (explore), low later (exploit)
        agent.ent_coef = ENT_START + (ENT_END - ENT_START) * episode / max(n_episodes - 1, 1)

        do_record = (cfg["video_every"] > 0 and episode % cfg["video_every"] == 0)

        # Save a greedy clip BEFORE this training episode so the video reflects
        # the policy at exactly this checkpoint. record_clip() does its own
        # env.reset() internally and spans as many episodes as needed to fill
        # VIDEO_MIN_STEPS — so the video is always full length even if the
        # agent dies quickly. The env is left open; we reset again below for
        # the training episode itself.
        if do_record:
            out_path = os.path.join(TRAIN_VID_DIR, f"train_ep{episode:05d}.mp4")
            record_clip(agent, env, cfg["frame_stack"],
                        VIDEO_MIN_STEPS, out_path, greedy=True)

        raw_obs, _ = env.reset()   # ← reset only, never close/reopen

        if not obs_vis_saved:
            save_obs_and_actions(raw_obs, n_actions,
                                 os.path.join(PLOT_DIR, "obs_and_actions.png"))
            obs_vis_saved = True

        stacker = FrameStack(cfg["frame_stack"], raw_obs.shape)
        obs     = stacker.reset(raw_obs)

        done, score, ep_steps = False, 0.0, 0

        while not done:
            action, logprob, val = agent.choose_action(obs, greedy=False)
            raw_obs_, reward, terminated, truncated, _ = env.step(action)
            done      = terminated or truncated
            n_steps  += 1
            ep_steps += 1
            score    += reward

            next_obs = stacker.step(raw_obs_)
            agent.remember(obs, action, logprob, val, reward, done)

            if n_steps % T == 0:
                agent.learn()
                learn_iters += 1

            obs = next_obs

        score_history.append(score)
        survival_history.append(ep_steps)
        timestep_history.append(n_steps)
        avg_score = float(np.mean(score_history[-100:]))

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        vid_tag = "  [VID]" if do_record else ""
        print(
            f"ep {episode:5d} | score {score:7.1f} | avg100 {avg_score:7.1f} | "
            f"best {best_score:7.1f} | survived {ep_steps:5d} | "
            f"steps {n_steps:9d} | updates {learn_iters} | "
            f"ent {agent.ent_coef:.4f}{vid_tag}"
        )

    save_reward_plot(score_history, timestep_history,
                     os.path.join(PLOT_DIR, "reward_vs_timesteps.png"))
    save_survival_plot(survival_history, timestep_history,
                       os.path.join(PLOT_DIR, "survival_vs_timesteps.png"))

    # Return env WITHOUT closing — main will pass it to evaluate()
    return agent, env, score_history, survival_history, n_actions


# ===========================================================================
# Evaluation
# ===========================================================================
def evaluate(agent, env, cfg, record_video=True):
    """
    Accepts an already-open env — does NOT create a new one.
    This is the Windows OpenGL fix: only one gym context ever exists.
    Caller is responsible for closing the env after this returns.

    Video strategy:
      Each eval episode is scored individually (for the bar chart).
      A single combined video is saved that stitches all eval episodes back
      to back, ensuring it is always long enough to be watchable even if the
      early policy dies quickly. The clip length is max(eval episodes total
      steps, VIDEO_MIN_STEPS) so you always get at least 40 seconds.
    """
    n_episodes  = cfg["eval_episodes"]
    frame_stack = cfg["frame_stack"]

    print(f"\n{'=' * 62}")
    print(f"  EVALUATION — {n_episodes} episodes | greedy (deterministic) policy")
    print(f"{'=' * 62}")

    # ---- Score each eval episode individually (no video here) ---------------
    eval_scores = []
    for ep in range(n_episodes):
        raw_obs, _ = env.reset()
        stacker    = FrameStack(frame_stack, raw_obs.shape)
        obs        = stacker.reset(raw_obs)

        done, score = False, 0.0
        while not done:
            action, _, _ = agent.choose_action(obs, greedy=True)
            raw_obs_, reward, terminated, truncated, _ = env.step(action)
            done   = terminated or truncated
            score += reward
            obs    = stacker.step(raw_obs_)

        eval_scores.append(score)
        print(f"    eval ep {ep:3d}  |  score {score:8.2f}")

    mean_s = float(np.mean(eval_scores))
    std_s  = float(np.std(eval_scores))
    print(f"\n  Mean +/- Std  :  {mean_s:.2f} +/- {std_s:.2f}")

    # ---- Save one combined video that is guaranteed to be full length --------
    if record_video:
        os.makedirs(EVAL_VID_DIR, exist_ok=True)
        vid_path  = os.path.join(EVAL_VID_DIR, "eval_combined.mp4")
        # Capture at least VIDEO_MIN_STEPS steps so the video is always watchable
        clip_steps = max(VIDEO_MIN_STEPS, n_episodes * 200)
        print(f"\n  Recording eval video ({clip_steps} steps / "
              f"{clip_steps/VIDEO_FPS:.0f}s)...")
        record_clip(agent, env, frame_stack, clip_steps, vid_path, greedy=True)

    return eval_scores


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    args = parse_args()
    cfg  = dict(
        lr            = args.lr,
        batch_size    = args.batch_size,
        n_epochs      = args.n_epochs,
        n_episodes    = args.n_episodes,
        gamma         = args.gamma,
        policy_clip   = args.policy_clip,
        frame_stack   = args.frame_stack,
        eval_episodes = args.eval_episodes,
        video_every   = args.video_every,
    )

    if args.eval_only:
        # ------------------------------------------------------------------
        # Eval-only: create the env ONCE, get n_actions from it, pass it in.
        # NO probe env — that was the crash trigger.
        # ------------------------------------------------------------------
        chk = os.path.join(CHKPT_DIR, "actor_critic_ppo.pt")
        if not os.path.exists(chk):
            raise FileNotFoundError(
                f"No checkpoint found at '{chk}'. "
                "Run without --eval-only to train first."
            )
        print("  [env] creating evaluation environment...")
        env       = gym.make(ENV_ID, render_mode="rgb_array")
        n_actions = env.action_space.n

        agent = Agent(n_actions=n_actions, in_channels=3 * cfg["frame_stack"])
        agent.load_models()

        eval_scores = evaluate(agent, env, cfg, record_video=True)
        env.close()   # ← only close here, at the very end

        save_eval_plot(eval_scores, os.path.join(PLOT_DIR, "eval_scores.png"))

    else:
        # ------------------------------------------------------------------
        # Full training + eval: train() returns env without closing it;
        # evaluate() reuses the same env; env.close() called exactly once.
        # ------------------------------------------------------------------
        agent, env, score_history, survival_history, n_actions = train(cfg)

        # Reload best checkpoint (may differ from final weights)
        agent.load_models()

        os.makedirs(EVAL_VID_DIR, exist_ok=True)
        eval_scores = evaluate(agent, env, cfg, record_video=True)
        env.close()   # ← single close, after all training and eval is done

        save_eval_plot(eval_scores, os.path.join(PLOT_DIR, "eval_scores.png"))
        save_summary(eval_scores, score_history, survival_history, cfg,
                     os.path.join(RESULTS_DIR, "summary.txt"))