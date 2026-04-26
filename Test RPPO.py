"""
Recurrent PPO (RPPO) for MiniWorld-CollectHealth-v0  —  FINAL FIX
==================================================================
ROOT CAUSE CONFIRMED:
  Health decreases inside the MiniWorld source at a fixed rate per step.
  Patching agent.health AFTER step() is too late — MiniWorld checks
  health <= 0 and sets terminated=True inside its own step() before
  our wrapper can restore any health. The agent dies at step 50
  regardless of what our wrapper does afterward.

SOLUTION:
  Subclass CollectHealth directly and override step() to reduce
  the health depletion rate before MiniWorld's termination check runs.
  Register it as a custom gymnasium environment.
  This is the ONLY reliable way to change the depletion rate.

Compatible with miniworld==2.1.0, gymnasium==0.29.1, sb3-contrib==2.3.0

Run:  python Recurrent_PPO.py
View: tensorboard --logdir logs/rppo_collecthealth
"""
from __future__ import annotations
import os

if "DISPLAY" not in os.environ:
    try:
        from pyvirtualdisplay import Display
        _display = Display(visible=False, size=(640, 480))
        _display.start()
        print("Virtual display started (headless mode)")
    except ImportError:
        print("No DISPLAY. Run: Xvfb :1 -screen 0 1024x768x24 & export DISPLAY=:1")

import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import miniworld
from miniworld.envs.collecthealth import CollectHealth
import cv2
import multiprocessing

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecTransposeImage, SubprocVecEnv, DummyVecEnv, VecFrameStack
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)


N_CPU = multiprocessing.cpu_count()
torch.set_num_threads(N_CPU)
print(f"PyTorch using {N_CPU} CPU cores")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────
# 1. Custom CollectHealth subclass with slower health depletion
#    This is the CORRECT place to fix health rate — inside step()
#    before MiniWorld checks if health <= 0 and terminates.
# ─────────────────────────────────────────────────────────────────────
class CollectHealthSlow(CollectHealth):
    """
    Slow down health depletion in CollectHealth by scaling per-step decay.

    Example:
      depletion_factor=0.25 means health decreases at 25% of default speed.
    """

    def __init__(self, *args, depletion_factor: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        if not (0.0 < depletion_factor <= 1.0):
            raise ValueError("depletion_factor must be in (0, 1].")
        self.depletion_factor = depletion_factor
        # CollectHealth uses a fixed per-step decay. We learn it online from
        # actual transitions to avoid hard-coding version-specific constants.
        self._estimated_decay = 1.0

    def _get_health(self):
        if hasattr(self, "agent") and self.agent is not None and hasattr(self.agent, "health"):
            return float(self.agent.health), ("agent", "health")
        if hasattr(self, "health"):
            return float(self.health), ("env", "health")
        return None, None

    def _set_health(self, value: float, health_ref):
        if health_ref == ("agent", "health"):
            self.agent.health = float(value)
        elif health_ref == ("env", "health"):
            self.health = float(value)

    def step(self, action):
        # --- Critical part: compensate BEFORE parent step/termination logic ---
        health_before, health_ref = self._get_health()
        if health_before is not None and health_ref is not None:
            # If default decay is D and we want factor f, add back (1-f)*D.
            # Parent step then subtracts D and checks death using this boosted
            # health, yielding an effective net decay of f*D.
            boosted = health_before + (1.0 - self.depletion_factor) * self._estimated_decay
            self._set_health(boosted, health_ref)
            health_before = boosted
        else:
            health_ref = None

        obs, reward, terminated, truncated, info = super().step(action)

        # Update decay estimate from transitions where health decreased.
        if health_before is not None and health_ref is not None:
            health_after, _ = self._get_health()
            if health_after is None:
                return obs, reward, terminated, truncated, info
            dec = health_before - health_after
            if dec > 0:
                self._estimated_decay = dec

        return obs, reward, terminated, truncated, info


register(
    id="MiniWorld-CollectHealthSlow-v0",
    entry_point=lambda **kwargs: CollectHealthSlow(**kwargs),
    max_episode_steps=2000,
)


def make_test_env(size: int = 10) -> gym.Env:
    return gym.make("MiniWorld-CollectHealthSlow-v0", render_mode=None, size=size)



# ─────────────────────────────────────────────────────────────────────
# 2. Action wrapper
# ─────────────────────────────────────────────────────────────────────
class ActionWrapper(gym.ActionWrapper):
    USEFUL_ACTIONS = [0, 1, 2]

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(self.USEFUL_ACTIONS))

    def action(self, act):
        return self.USEFUL_ACTIONS[act]


# ─────────────────────────────────────────────────────────────────────
# 3. Health-kit reward wrapper
#    +50 for collecting a kit, +0.5 for moving forward
# ─────────────────────────────────────────────────────────────────────
class HealthKitRewardWrapper(gym.Wrapper):
    KIT_BONUS     = 50.0
    FORWARD_BONUS = 0.5

    def __init__(self, env):
        super().__init__(env)
        self._last_health = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_health = self._get_health()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_health = self._get_health()

        if self._last_health is not None and current_health > self._last_health:
            reward += self.KIT_BONUS

        if action == 2:
            reward += self.FORWARD_BONUS

        self._last_health = current_health
        info["health"]    = current_health
        return obs, reward, terminated, truncated, info

    def _get_health(self):
        base = self.env.unwrapped
        if hasattr(base, "agent") and base.agent and hasattr(base.agent, "health"):
            return base.agent.health
        return 10000.0


# ─────────────────────────────────────────────────────────────────────
# 4. Grayscale wrapper
# ─────────────────────────────────────────────────────────────────────
class GrayWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]


# ─────────────────────────────────────────────────────────────────────
# 5. Environment factory
# ─────────────────────────────────────────────────────────────────────
def make_env(rank=0, seed=0, reward_shaping=True, room_size=10):
    def _init():
        # Use our custom slow-health environment
        env = gym.make(
            "MiniWorld-CollectHealthSlow-v0",
            render_mode = None,
            size        = room_size,
        )
        env = ActionWrapper(env)
        if reward_shaping:
            env = HealthKitRewardWrapper(env)
        env = GrayWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────
# 6. Learning rate schedule
# ─────────────────────────────────────────────────────────────────────
def linear_lr_schedule(initial_lr: float, final_lr: float):
    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)
    return schedule


# ─────────────────────────────────────────────────────────────────────
# 7. Stats callback
# ─────────────────────────────────────────────────────────────────────
class StatsCallback(BaseCallback):
    def __init__(self, reward_threshold=100.0):
        super().__init__()
        self.ep_rewards       = []
        self.ep_lengths       = []
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r   = info["episode"]["r"]
                l   = info["episode"]["l"]
                self.ep_rewards.append(r)
                self.ep_lengths.append(l)
                avg     = np.mean(self.ep_rewards[-100:])
                avg_len = np.mean(self.ep_lengths[-100:])

                self.logger.record("custom/ep_reward",     r)
                self.logger.record("custom/ep_length",     l)
                self.logger.record("custom/avg100_reward", avg)
                self.logger.record("custom/avg100_length", avg_len)
                self.logger.dump(self.num_timesteps)

                print(f"  [ep {len(self.ep_rewards):5d} | "
                      f"step {self.num_timesteps:8d}]  "
                      f"reward={r:8.1f}  len={l:5d}  "
                      f"avg100={avg:8.1f}")

                if avg >= self.reward_threshold and len(self.ep_rewards) >= 100:
                    print(f"\nSolved! avg100={avg:.1f}")
                    return False
        return True


# ─────────────────────────────────────────────────────────────────────
# 8. Hyperparameters
# ─────────────────────────────────────────────────────────────────────
N_ENVS          = 8
TOTAL_TIMESTEPS = 5_000_000
N_STACK         = 4

RPPO_PARAMS = dict(
    learning_rate   = linear_lr_schedule(3e-4, 5e-5),
    n_steps         = 512,
    batch_size      = 128,
    n_epochs        = 4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.10,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    verbose         = 1,
    tensorboard_log = "logs/rppo_collecthealth",
    policy_kwargs   = dict(
        lstm_hidden_size   = 256,
        n_lstm_layers      = 1,
        shared_lstm        = True,
        enable_critic_lstm = False,
    ),
)


if __name__ == "__main__":

    # ── Verify the custom environment works ───────────────────────
    print("Verifying custom slow-health environment...")
    _test = gym.make(
        "MiniWorld-CollectHealthSlow-v0",
        render_mode=None,
        size=10,
    )

    _obs, _ = _test.reset()

    print(f"  Obs shape      : {_obs.shape}")
    print(f"  Action space   : {_test.action_space}")

    health = getattr(_test.unwrapped.agent, "health", 10000.0)
    print(f"  Initial health : {health:.1f}")
    # Walk forward and check episode length
    for _s in range(500):
        _obs, _r, _term, _trunc, _ = _test.step(2)
        if _term or _trunc:
            print(f"  Agent died at step {_s+1} with slow health")
            break
    else:
        print(f"  Survived 500 steps! Slow health is working.")
    _test.close()
    print("  Environment OK\n")

    # ── Build environments ────────────────────────────────────────
    vec_env = make_vec_env(
        make_env(reward_shaping=True, room_size=10),
        n_envs      = N_ENVS,
        vec_env_cls = SubprocVecEnv,
    )
    vec_env = VecTransposeImage(vec_env)
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)

    eval_env = make_vec_env(
        make_env(seed=999, reward_shaping=False, room_size=10),
        n_envs      = 1,
        vec_env_cls = SubprocVecEnv,
    )
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

    # ── Callbacks ─────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = "models/best_rppo",
        log_path             = "logs/rppo_collecthealth/eval",
        eval_freq            = 25_000,
        n_eval_episodes      = 10,
        deterministic        = True,
        render               = False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = 100_000,
        save_path   = "models/checkpoints_rppo",
        name_prefix = "rppo_collecthealth",
    )

    stats_callback = StatsCallback(reward_threshold=100.0)

    # ── Model ─────────────────────────────────────────────────────
    model = RecurrentPPO(
        policy = "CnnLstmPolicy",
        env    = vec_env,
        **RPPO_PARAMS,
    )

    print("Policy architecture:")
    print(model.policy)
    print(f"\nObs shape     : {vec_env.observation_space.shape}")
    print(f"Action space  : {vec_env.action_space}")
    print(f"Parallel envs : {N_ENVS}")
    print(f"Frame stack   : {N_STACK}")
    print(f"Data/update   : {N_ENVS * RPPO_PARAMS['n_steps']} transitions\n")

    # ── Train ─────────────────────────────────────────────────────
    print("Starting training...")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [eval_callback, checkpoint_callback, stats_callback],
        progress_bar    = True,
    )

    model.save("models/rppo_collecthealth_final")
    print("Training complete.")

    # ── Final evaluation ──────────────────────────────────────────
    print("\n===== FINAL EVALUATION (10 episodes) =====")

    best_model = RecurrentPPO.load("models/best_rppo/best_model", env=eval_env)
    test_scores = []

    for ep in range(10):
        obs            = eval_env.reset()
        total_reward   = 0.0
        done           = False
        lstm_states    = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            action, lstm_states = best_model.predict(
                obs,
                state         = lstm_states,
                episode_start = episode_starts,
                deterministic = True,
            )
            obs, reward, done, info = eval_env.step(action)
            total_reward  += reward[0]
            episode_starts = np.zeros((1,), dtype=bool)

        test_scores.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.1f}")

    print(f"\nMean reward : {np.mean(test_scores):.1f}")
    print(f"Std         : {np.std(test_scores):.1f}")

    # ── Video recording ───────────────────────────────────────────
    print("\nRecording demo video...")

    video_env = gym.make(
        "MiniWorld-CollectHealthSlow-v0",
        render_mode = "rgb_array",
        size        = 10,
    )
    video_env = ActionWrapper(video_env)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter("rppo_demo.mp4", fourcc, 30.0, (80, 60))

    video_model    = RecurrentPPO.load("models/best_rppo/best_model")
    frame_buf      = None

    def get_stacked_obs(rgb):
        global frame_buf
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        chw  = gray.transpose(2, 0, 1)
        if frame_buf is None:
            frame_buf = np.repeat(chw, N_STACK, axis=0)
        else:
            frame_buf = np.concatenate([frame_buf[1:], chw], axis=0)
        return frame_buf[np.newaxis]

    raw_obs, _     = video_env.reset(seed=42)
    lstm_states    = None
    episode_starts = np.ones((1,), dtype=bool)
    total_reward   = 0.0
    episodes_done  = 0

    for step in range(6000):
        model_obs = get_stacked_obs(raw_obs)
        action, lstm_states = video_model.predict(
            model_obs,
            state         = lstm_states,
            episode_start = episode_starts,
            deterministic = True,
        )

        raw_obs, reward, term, trunc, _ = video_env.step(action[0])
        total_reward += reward

        frame = video_env.render()
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        episode_starts = np.array([term or trunc], dtype=bool)

        if term or trunc:
            episodes_done += 1
            print(f"  Video ep {episodes_done}: steps={step+1}, "
                  f"reward={total_reward:.1f}")
            total_reward = 0.0
            raw_obs, _   = video_env.reset()
            frame_buf    = None
            lstm_states  = None
            if episodes_done >= 3:
                break

    out.release()
    video_env.close()
    vec_env.close()
    eval_env.close()
    print("Video saved → rppo_demo.mp4")