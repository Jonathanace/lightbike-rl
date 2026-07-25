import streamlit as st
import numpy as np
from PIL import Image
import glob
from stable_baselines3 import PPO
from lightbike_rl import lightbike_v0

# Absolute mapping matching DIR_TO_INT (U=0, R=1, D=2, L=3)
ABSOLUTE_MAP = {
    "W": 0,
    "D": 1,
    "S": 2,
    "A": 3
}

HUMAN_AGENT_ID = "player_0"

def load_policies():
    return list(glob.glob("*.zip"))

def render_grid_to_image(grid):
    rgb_array = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    rgb_array[grid == 0] = [10, 10, 25]
    rgb_array[grid == -1] = [100, 100, 100]
    rgb_array[grid == 1] = [0, 255, 255]
    rgb_array[grid == 2] = [255, 0, 255]
    img = Image.fromarray(rgb_array)
    img = img.resize((500, 500), Image.NEAREST)
    return img

def main():
    st.title("Play Lightbike-RL")
    policy = st.selectbox("Select opponent policy", load_policies(), key='select_policy')
    st.button('Start/Reset', on_click=init_game, args=[policy])

    # UI
    if 'message' in st.session_state and st.session_state.message:
        if st.session_state.message_type == 'warning':
            st.warning(st.session_state.message)
        elif st.session_state.message_type == 'success':
            st.success(st.session_state.message)
        st.session_state.message = None

    if 'game_img' in st.session_state and st.session_state.game_img is not None:
        st.image(st.session_state.game_img, use_column_width=True)
    else:
        st.info("Press Start/Reset to begin!")

    # Controls
    st.write("### Controls")
    cols = st.columns(4)
    with cols[0]:
        st.button("Up (W)", on_click=step_game, args=["W"], use_container_width=True, shortcut="W")
    with cols[1]:
        st.button("Left (A)", on_click=step_game, args=["A"], use_container_width=True, shortcut="A")
    with cols[2]:
        st.button("Down (S)", on_click=step_game, args=["S"], use_container_width=True, shortcut="S")
    with cols[3]:
        st.button("Right (D)", on_click=step_game, args=["D"], use_container_width=True, shortcut="D")

def init_game(policy):
    st.session_state.model = PPO.load(policy)
    st.session_state.env = lightbike_v0.parallel_env()
    st.session_state.observations, infos = st.session_state.env.reset()
    base_env = st.session_state.env.unwrapped
    st.session_state.game_img = render_grid_to_image(base_env.grid)
    st.session_state.game_active = True
    st.session_state.message = None

def step_game(user_input):
    if not st.session_state.get('game_active', False):
        st.session_state.message = "Start the game first!"
        st.session_state.message_type = "warning"
        return

    env = st.session_state.env
    model = st.session_state.model
    obs = st.session_state.observations
    base_env = env.unwrapped

    if not env.agents:
        st.session_state.game_active = False
        return

    # Get relative direction from asbolute input
    human_idx = int(HUMAN_AGENT_ID.split("_")[1])
    current_heading = base_env.current_dirs[human_idx]
    target_heading = ABSOLUTE_MAP[user_input]

    # Calculate the turning difference
    diff = (target_heading - current_heading) % 4

    if diff == 0:
        human_relative_action = 0 # Forward
    elif diff == 1:
        human_relative_action = 2 # Right
    elif diff == 3:
        human_relative_action = 1 # Left
    elif diff == 2:
        # Invalid 180-degree turn. Default to moving forward.
        human_relative_action = 0
        st.session_state.message = "Can't reverse! Moving forward."
        st.session_state.message_type = "warning"

    # Assign actions to all agents
    actions = {}
    for agent in env.agents:
        if agent == HUMAN_AGENT_ID:
            actions[agent] = human_relative_action
        else:
            action, _states = model.predict(obs[agent], deterministic=True)
            actions[agent] = action.item()

    # Step the environment
    st.session_state.observations, rewards, terminations, truncations, infos = env.step(actions)
    st.session_state.game_img = render_grid_to_image(base_env.grid)

    if not env.agents:
        st.session_state.game_active = False
        winner_msg = f"Game Over! Winner: {base_env.winner}" if base_env.winner else "Game Over! Tie."
        st.session_state.message = winner_msg
        st.session_state.message_type = "success"

if __name__ == "__main__":
    main()