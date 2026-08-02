from gymnasium.spaces import Space, Dict, Discrete
import gzip
from datetime import date
from pettingzoo.test import parallel_api_test
from pettingzoo import ParallelEnv
from typing import NamedTuple
from collections import defaultdict
from lightbike_rl.utils import render, save_frame
import logging
import json
import uuid
import os
import gymnasium as gym
import numpy as np
from dataclasses import dataclass

@dataclass
class EnvParams:
    x_size: int = 51
    y_size: int = 51
    x_pos = x_size//2+1
    y_diff = 10

    # Spawn Settings
    starting_pos: tuple = ((x_pos, y_diff), (x_pos, y_size-y_diff)) # [row, col]
    starting_dirs: tuple = ("R", "L")

    max_steps: int = 1000
    # max_sensor_range: int = -1

    num_players = 2
@dataclass
class WorkerConfig:
    log_dir: str = "replays"
class DirPayload(NamedTuple):
    name: str
    idx: int
    coords: tuple[int, int]

def create_dir_map(directions_list):
    mapping = {}
    for auto_idx, (name, coords) in enumerate(directions_list):
        payload = DirPayload(name=name, idx=auto_idx, coords=coords)

        mapping[name] = payload
        mapping[auto_idx] = payload
        mapping[coords] = payload

    return mapping

DIR_TO_INT = {"U": 0, "R": 1, "D": 2, "L": 3}

directions = [
    ("U",  (-1, 0)),
    ("R",  (0, 1)),
    ("D",  (1, 0)),
    ("L",  (0, -1)),
]

DIR_MAP = create_dir_map(directions)
class LightBikeEnv(ParallelEnv):
    _distance_obs = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    metadata = {
        "name": "liightbike_v0"
    }

    def __init__(self, config=None, env_config=None, debug=False):
        # Params
        env_config = env_config or {}
        self.params = EnvParams(**env_config)

        config = config or {}
        self.config = WorkerConfig(**config)
        self.render_mode=None

        # Agents
        self.agents = self.possible_agents = [
            f"player_{i}" for i in range(self.params.num_players)
        ]
        self.num_players = len(self.possible_agents)
        self.alive = [1] * len(self.agents)

        # Starting parameters
        self.starting_positions = np.array(self.params.starting_pos, dtype=np.int32)
        self.positions = np.empty_like(self.starting_positions)

        # Starting grid
        self.starting_grid = np.zeros((self.params.y_size, self.params.x_size), dtype=np.int32)
        for player_i in range(self.num_players):
            pos = tuple(self.starting_positions[player_i])
            self.starting_grid[pos] = player_i + 1
            self.positions[player_i] = pos

        self.starting_grid = np.pad(
            self.starting_grid, pad_width=1, mode='constant', constant_values=255
        )
        self.grid = np.empty_like(self.starting_grid)
        self.starting_dirs_int = np.array([DIR_TO_INT[dir] for dir in self.params.starting_dirs], dtype=np.int32)
        self.current_dirs = np.empty_like(self.starting_dirs_int)
        self.starting_positions += 1

        self.winner = None
        self.ended = False
        self.debug = debug
        self.episode = defaultdict(list)

        # self.dict_space: dict[str, Space] = {
        #         # Distances to nearest wall
        #         "distances": gym.spaces.Box(
        #             low=0,
        #             high=1,
        #             shape=(self.num_players, len(directions)),
        #             dtype=np.float32
        #         ),

        #         # Player positions
        #         "positions": gym.spaces.Box(
        #             low=0,
        #             high=1,
        #             shape=(self.num_players, 2),
        #             dtype=np.float32
        #         ),

        #         # Distances to players
        #         "pos_diff": gym.spaces.Box(
        #             low=-1,
        #             high=1,
        #             shape=(self.num_players, self.num_players),
        #             dtype=np.float32
        #         )
        #     }
        self.obs_space = gym.spaces.Box(low=0, high=255,
                                            shape=(3, self.starting_grid.shape[0], self.starting_grid.shape[1]), dtype=np.uint8)
        self.act_space = Discrete(4)

        if self.debug:
            logging.debug("Debug mode activated.")

    def reset(self, seed=None, options=None):
        logging.debug("Resetting environment...")
        self._reset_game()
        obs = self._get_all_obs()
        return obs, {a: {} for a in self.agents}

    def step(self, actions=None):
        logging.debug("Step called")
        if not actions:
            logging.debug("No action dict passed. Selecting random actions.")
            actions = {player: np.random.randint(0, 4) for player in self.agents}

        action_names = {player: DIR_MAP[action].name for player, action in actions.items()}
        logging.debug(f"Actions: {action_names}")

        for player in list(self.agents):
            player_idx = int(player.split("_")[1])
            action = actions[player]
            self._step_player(player_idx, action)

        observations = self._get_all_obs()

        if sum(self.alive) == 0:
            logging.debug("All players lost, game is a tie")
            self.winner = None
            self.ended = True
        elif sum(self.alive) == 1:
            win_player_idx = self.alive.index(1)
            self.winner = f"player_{win_player_idx}"
            logging.debug(f"{self.winner} wins!")
            self.ended = True

        if self.ended:
            # self.save_replay()
            pass

        terminations = {a: self.ended for a in self.agents}

        rewards = {}
        for a in self.agents:
            p_idx = int(a.split("_")[1])
            if not self.ended:
                rewards[a] = 0.0
            elif a == self.winner:
                rewards[a] = 1.0
            elif self.alive[p_idx] == 0:
                rewards[a] = -1.0
            else:
                rewards[a] = 0.0  # Draw condition

        truncated = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        self.agents = [a for a in self.agents if not terminations[a]]

        return (
            observations,
            rewards,
            terminations,
            truncated,
            infos
        )

    def render(self):
        render(self.grid)

    def save_frame(self):
        save_frame(self.grid)

    def observation_space(self, agent):
        return self.obs_space

    def action_space(self, agent):
        return self.act_space

    def _reset_game(self):
        self.agents = self.possible_agents
        np.copyto(self.positions, self.starting_positions)
        np.copyto(self.grid, self.starting_grid)
        np.copyto(self.current_dirs, self.starting_dirs_int)

    def _get_all_obs(self):
        return {
            agent: self._get_player_obs(agent_i) for agent_i, agent in enumerate(self.agents)
        }

    def _get_player_obs(self, player_idx: int) -> np.ndarray:
        obs = np.zeros((3, self.grid.shape[0], self.grid.shape[1]), dtype=np.uint8)

        my_id = player_idx + 1
        opp_id = 2 if my_id == 1 else 1

        # Current player head
        obs[0] = (self.grid == my_id) * 255

        opps_ids = [i+1 for i in range(0, self.num_players) if i != player_idx]
        obs[1] = np.isin(self.grid, opps_ids) * 255

        obs[2] = (self.grid == 255) * 255

        return obs

    # @property
    # def distances(self):
    #     return [self._get_distance(player_i) for player_i in range(self.num_players)]

    # @property
    # def pos_diff(self):
    #     abs_diff = np.abs(self.positions[:, None, :] - self.positions[None, :, :])
    #     manhattan_matrix = np.sum(abs_diff, axis=-1)
    #     return manhattan_matrix.astype(np.float32)

    # def _get_distance(self, player_i):
    #     y, x = self.positions[player_i]
    #     distances = []
    #     for dir_i in range(len(directions)):
    #         dy, dx = DIR_MAP[dir_i].coords
    #         steps = 1
    #         while True:
    #             target_x = x + steps * dx
    #             target_y = y + steps * dy
    #             target = self.grid[target_y, target_x]
    #             if target != 0:
    #                 distances.append(steps)
    #                 break
    #             steps += 1
    #             if steps > max(self.grid.shape):
    #                 error = f"y: {target_y}, x: {target_x} is out of bounds for grid dimensions {self.grid.shape}"
    #                 raise ValueError(error)
    #     return distances

    # def _localize_obs(self, obs, idx):
    #     return np.roll(obs, shift=-idx, axis=0)

    # def _normalize(self, obs, obs_type):
    #     max_y = self.params.y_size
    #     max_x = self.params.x_size

    #     match obs_type:
    #         case "distances":
    #             max_dim = max(max_y, max_x)
    #             return np.array(obs, dtype=np.float32) / max_dim

    #         case "positions":
    #             scale_factors = np.array([max_y, max_x], dtype=np.float32)
    #             return np.array(obs, dtype=np.float32) / scale_factors

    #         case "pos_diff":
    #             max_manhattan = max_y + max_x
    #             return np.array(obs, dtype=np.float32) / max_manhattan

    #         case _:
    #             raise ValueError(f"Invalid obs_type: {obs_type}")

    # def _get_dist(self, player_idx, target_direction):
    #     d_x, d_y = target_direction
    #     return

    def _step_player(self, player_i: int, action_num: int) -> tuple[float, bool]:
        # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        dy, dx = DIR_MAP[action_num].coords

        old_y, old_x = self.positions[player_i]
        new_y, new_x = old_y + dy, old_x + dx

       # boundary check
        if new_y < 0 or new_y >= self.grid.shape[0] or new_x < 0 or new_x >= self.grid.shape[1]:
            self.alive[player_i] = 0
            return -1.0, True

        # Collision check
        if self.grid[new_y, new_x] != 0:
            self.alive[player_i] = 0
            return -1.0, True

        # Update state
        self.positions[player_i] = (new_y, new_x)
        self.grid[new_y, new_x] = player_i + 1
        self.grid[old_y, old_x] = 255

        return 0.0, False

    def save_replay(self):
        os.makedirs(self.config.log_dir, exist_ok=True)
        unique_id = uuid.uuid4()
        filename = str(date.today()) + str(unique_id) + ".json.gz"
        filepath = os.path.join(self.config.log_dir, filename)
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(self.episode, f)

    def sample(self, n=20):
        self.reset()
        render(self.grid)
        for i in range(n):
            self.step()
            if self.ended:
                break
            render(self.grid)
        self.reset()


def sample_env():
    env = LightBikeEnv()
    observations, infos = env.reset(seed=42)
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(actions)
        env.render_in_terminal()
        input("Press any key to continue...")
    env.close()

def test_env():
    logging.basicConfig(level=logging.DEBUG)
    env = LightBikeEnv()
    env.sample()

    parallel_api_test(env, num_cycles=100)

if __name__ == "__main__":
    test_env()
    sample_env()