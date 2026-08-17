# Overview
This is a multi-agent Lightbike (Tron) reinforcement learning environment. Created using [Farama PettingZoo](https://pettingzoo.farama.org/index.html). Includes training scripts and wandb monitoring. 

The environment definition can be found at `lightbike_rl/env/lightbike.py`, and the training scripts can be found in `scripts/`.  

# Usage
1. Run `uv sync` to setup the virtual environment
2. Run `uv run wandb login` to authenticate your wandb account.
3. Run `uv run scripts/train_sb3.py` to start the training loop.
4. Run `streamlit run game.py` to play against your trained policy. 

# Coming Soon
1. Docker Support.
2. Full frame-by-frame replay rendering.
3. Allow users to play against the trained policy.
4. Policy training checkpointing.
5. Ray/Rllib training implementation. 
