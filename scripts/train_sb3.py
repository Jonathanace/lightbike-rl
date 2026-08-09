import logging
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy, CnnPolicy
from lightbike_rl import lightbike_v0

import wandb
from wandb.integration.sb3 import WandbCallback

def _train(policy_n: int | None = None):
    if policy_n is None:
        policies = load_policies()
        reserved_ns = [int(policy[-5]) for policy in policies]
        policy_n = 1
        while policy_n in reserved_ns:
            policy_n += 1
    run = wandb.init(
        project="lightbike-rl",
        sync_tensorboard=True,
        monitor_gym=False,
    )

    env = lightbike_v0.parallel_env()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class='stable_baselines3')

    model = PPO(
        CnnPolicy,
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
        total_timesteps=2_000_000,
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=500_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    )

    model.save("policy_5")

    run.finish()

def _play():
    env = lightbike_v0.parallel_env()
    model = PPO.load("policy_6")
    observations, infos = env.reset()
    while env.agents:
            actions = {}
            # env.save_frame() # can't do this on headless

            for agent in env.agents:
                action, _states = model.predict(observations[agent], deterministic=False)
                actions[agent] = action.item()

            observations, rewards, terminations, truncations, infos = env.step(actions)
            input()

    env.close()

if __name__ == "__main__":
    _train()
    # _play()
