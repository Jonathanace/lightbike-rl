import logging
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from lightbike_rl import lightbike_v0

import wandb
from wandb.integration.sb3 import WandbCallback

def _train():
    run = wandb.init(
        project="lightbike-rl",
        sync_tensorboard=True,
        monitor_gym=False,
    )

    env = lightbike_v0.parallel_env()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class='stable_baselines3')

    model = PPO(
        MultiInputPolicy,
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        gamma=0.995,
        n_steps=1024,
        ent_coef=0.01,
        learning_rate=0.0003,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        batch_size=2048
    )

    model.learn(
        total_timesteps=5_000_000,
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )

    model.save("policy_3")

    run.finish()

def _play():
    env = lightbike_v0.parallel_env()
    model = PPO.load("policy_3")
    observations, infos = env.reset()
    while env.agents:
            actions = {}
            # env.save_frame() # can't do this on headless

            for agent in env.agents:
                action, _states = model.predict(observations[agent], deterministic=True)
                actions[agent] = action.item()

            observations, rewards, terminations, truncations, infos = env.step(actions)
            input()

    env.close()

if __name__ == "__main__":
    _train()
    # _play()
