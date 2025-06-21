import yaml
import wandb
import os
import msvcrt  # For Windows keypress detection
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium import make as gym_make
from soulsai.wrappers.iudex_wrappers import IudexObservationWrapper
from stable_baselines3.common.callbacks import BaseCallback
import uuid

# Custom callback to stop after N episodes
class StopAfterNEpisodesCallback(BaseCallback):
    def __init__(self, max_episodes, verbose=0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            if any(self.locals["dones"]):
                self.episode_count += sum(self.locals["dones"])
                # Only stop if max_episodes > 0
                if self.max_episodes > 0 and self.episode_count >= self.max_episodes:
                    print(f"Reached {self.episode_count} episodes, stopping training.")
                    return False
        return True

# Custom callback to log average episode boss HP to wandb
class LogBossHPCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_boss_hp = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "boss_hp" in info:
                self.episode_boss_hp.append(info["boss_hp"])
        dones = self.locals.get("dones", [])
        if any(dones):
            if self.episode_boss_hp:
                avg_boss_hp = sum(self.episode_boss_hp) / len(self.episode_boss_hp)
                wandb.log({"episode_boss_hp": avg_boss_hp, "global_step": self.num_timesteps})
                self.episode_boss_hp = []
        return True

# Custom callback for periodic checkpointing every N episodes (epochs)
class PeriodicCheckpointCallback(BaseCallback):
    def __init__(self, save_freq_episodes, model_path, replay_buffer_path, verbose=0):
        super().__init__(verbose)
        self.save_freq_episodes = save_freq_episodes
        self.model_path = model_path
        self.replay_buffer_path = replay_buffer_path
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        if any(dones):
            self.episode_count += sum(dones)
            if self.episode_count % self.save_freq_episodes == 0:
                print(f"[Checkpoint] Saving model and replay buffer at episode {self.episode_count}")
                self.model.save(self.model_path)
                self.model.save_replay_buffer(self.replay_buffer_path)
        return True

# Custom callback to stop training when Enter is pressed (Windows only)
class StopOnEnterCallback(BaseCallback):
    def _on_step(self) -> bool:
        if msvcrt.kbhit() and msvcrt.getch() == b'\r':
            print("Enter pressed! Stopping training.")
            return False
        return True

# Custom callback to log episode wins to wandb
class LogWinsCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                win = 1 if info.get("boss_hp", 1) == 0 else 0
                wandb.log({"episode_win": win, "global_step": self.num_timesteps})
        return True

# 1. Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Extract environment and wandb settings
env_name = config['env']['name']
env_kwargs = config['env'].get('kwargs', {}) or {}
wandb_cfg = config.get('monitoring', {}).get('wandb', {})
max_episodes = config.get('max_episodes', 100)

# 3. Prepare model/replay buffer paths and wandb run ID file
checkpoint_dir = os.path.join("saves", "checkpoint")
os.makedirs(checkpoint_dir, exist_ok=True)
model_path = os.path.join(checkpoint_dir, "dqn_soulsgym_iudex")
replay_buffer_path = os.path.join(checkpoint_dir, "dqn_soulsgym_iudex_replay_buffer")
run_id_path = os.path.join(checkpoint_dir, "wandb_run_id.txt")
# For periodic checkpointing (only keep latest)
checkpoint_model_path = os.path.join(checkpoint_dir, "dqn_checkpoint_latest.zip")
checkpoint_replay_buffer_path = os.path.join(checkpoint_dir, "dqn_checkpoint_latest_replay_buffer.pkl")

# 4. Handle wandb run ID for resuming
if os.path.exists(run_id_path):
    with open(run_id_path, 'r') as f:
        wandb_run_id = f.read().strip()
    resume_mode = "allow"
    print(f"Resuming wandb run with ID: {wandb_run_id}")
else:
    wandb_run_id = str(uuid.uuid4())
    with open(run_id_path, 'w') as f:
        f.write(wandb_run_id)
    resume_mode = None
    print(f"Starting new wandb run with ID: {wandb_run_id}")

# 5. Initialize wandb (with run ID for resuming)
wandb.init(
    project=wandb_cfg.get('project', 'soulsai_iudex'),
    entity=wandb_cfg.get('entity'),
    group=wandb_cfg.get('group'),
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
    id=wandb_run_id,
    resume=resume_mode,
)

# 6. Create and wrap the environment
env = gym_make(env_name, **env_kwargs)
env = IudexObservationWrapper(env)
env = Monitor(env)

dqn_params = config.get('dqn', {})
learning_rate = dqn_params.get('agent', {}).get('kwargs', {}).get('lr', 1e-3)
gamma = dqn_params.get('agent', {}).get('kwargs', {}).get('gamma', 0.99)
batch_size = dqn_params.get('batch_size', 64)

# 7. Load model and replay buffer if they exist, else create new
# Prioritize latest checkpoint over main checkpoint
if os.path.exists(checkpoint_model_path):
    print("Resuming from latest checkpoint...")
    model = DQN.load(checkpoint_model_path, env=env)
    if os.path.exists(checkpoint_replay_buffer_path):
        model.load_replay_buffer(checkpoint_replay_buffer_path)
elif os.path.exists(model_path + ".zip"):
    print("Resuming from saved model and replay buffer...")
    model = DQN.load(model_path, env=env)
    if os.path.exists(replay_buffer_path + ".pkl"):
        model.load_replay_buffer(replay_buffer_path)
else:
    print("Starting new training run...")
    # Get tensorboard log directory safely
    tensorboard_log = wandb.run.dir if wandb.run and wandb.run.dir else None
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

# 8. Train the agent, stopping after max_episodes, logging boss HP, periodic checkpointing, and Enter key
model.learn(
    total_timesteps=int(1e10),  # Large value, episode callback will stop first
    callback=[
        StopAfterNEpisodesCallback(max_episodes),
        LogBossHPCallback(),
        LogWinsCallback(),  # Log episode wins
        PeriodicCheckpointCallback(
            save_freq_episodes=100,
            model_path=checkpoint_model_path,
            replay_buffer_path=checkpoint_replay_buffer_path
        ),
        StopOnEnterCallback(),  # Now supports stopping with Enter
    ],
)

# 9. Save the model and replay buffer at the end
model.save(model_path)
model.save_replay_buffer(replay_buffer_path)
wandb.save(model_path + ".zip")
wandb.save(replay_buffer_path + ".pkl")
wandb.save(checkpoint_model_path)
wandb.save(checkpoint_replay_buffer_path)

print("Training complete. Model, replay buffer, and latest checkpoint saved for automatic resume.")
