from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Space
from typing import NamedTuple
from collections import defaultdict
from lightbike_rl.utils import render
import random
import logging
import json
import uuid
import os
from numba import int32
from numba.experimental import jitclass
from enum import IntEnum
import gymnasium as gym
import numpy as np
from dataclasses import dataclass
from lightbike_rl.constants import COLOR_MAP
from gymnasium.spaces import flatten_space, flatten

@dataclass
class EnvParams:
    x_size: int = 101
    y_size: int = 101

    # Spawn Settings
    starting_pos: tuple = ((51, 10), (51, 90)) # [row, col]
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


directions = [
    ("L",  (0, -1)),
    ("R",  (0, 1)),
    ("U",  (-1, 0)),
    ("D",  (1, 0)),
    ("UL", (-1, -1)),
    ("UR", (-1, 1)),
    ("BL", (1, -1)),
    ("BR", (1, 1))
]

DIR_MAP = create_dir_map(directions)


class LightBikeEnv(MultiAgentEnv):
    _distance_obs = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    _ACTION_SPACE = gym.spaces.Discrete(3) # Left, right, forward

    def _get_observation_space(self) -> dict[str, Space]:
        return {
            # Distances to nearest wall
            "distances": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.num_players, len(directions)),
                dtype=np.float32
            ),
            # Player positions
            "positions": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.num_players, 2),
                dtype=np.float32
            ),

            "pos_diff": gym.spaces.Box(
                low=-1,
                high=1,
                shape=(self.num_players, self.num_players),
                dtype=np.float32
            )
        }



    def __init__(self, config=None, env_config=None, debug=False):
        super().__init__()

        # Params
        env_config = env_config or {}
        self.params = EnvParams(**env_config)

        config = config or {}
        self.config = WorkerConfig(**config)

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
            self.starting_grid, pad_width=1, mode='constant', constant_values=-1
        )
        self.grid = np.empty_like(self.starting_grid)
        self.starting_positions += 1

        self.internal_obs_space = gym.spaces.Dict(self._get_observation_space())

        self.flat_obs_space = flatten_space(self.internal_obs_space)

        self.observation_space = gym.spaces.Dict({
            str(agent): self.flat_obs_space for agent in self.agents
        })

        self.action_space = gym.spaces.Dict({
            str(agent): self._ACTION_SPACE for agent in self.agents
        })

        self.ended = False
        self.debug = debug
        self.episode = defaultdict(list)

        if self.debug:
            logging.debug("Debug mode activated.")


    def reset(self, *, seed=None, options=None):
        self._reset_game()
        return self._get_all_obs(), {}

    def _reset_game(self):
        np.copyto(self.positions, self.starting_positions)
        np.copyto(self.grid, self.starting_grid)

    def _get_all_obs(self):
        return {
            agent: self._get_player_obs(agent) for agent in self.agents
        }

    def _get_player_obs(self, player):
        player_idx = int(player.split("_")[1])
        obs = {}
        for i, obs_name in enumerate(self._get_observation_space().keys()):
            if not hasattr(self, obs_name):
                raise AttributeError(f"Attribute {obs_name} does not exist.")

            raw_obs = getattr(self, obs_name)

            if raw_obs is None:
                raise Exception(f"Observation {obs_name} is None.")

            normalized_obs = self._normalize(raw_obs, obs_name)
            localized_obs = self._localize_obs(normalized_obs, i)
            obs[obs_name] = localized_obs

        return flatten(self.internal_obs_space, obs)

    @property
    def distances(self):
        return [self._get_distance(player_i) for player_i in range(self.num_players)]

    @property
    def pos_diff(self):
        abs_diff = np.abs(self.positions[:, None, :] - self.positions[None, :, :])
        manhattan_matrix = np.sum(abs_diff, axis=-1)
        return manhattan_matrix.astype(np.float32)

    def _get_distance(self, player_i):
        y, x = self.positions[player_i]
        distances = []
        for dir_i in range(len(directions)):
            dy, dx = DIR_MAP[dir_i].coords
            steps = 1
            while True:
                target_x = x + steps * dx
                target_y = y + steps * dy
                target = self.grid[target_y, target_x]
                if target != 0:
                    distances.append(steps)
                    break
                steps += 1
                if steps > max(self.grid.shape):
                    error = f"y: {target_y}, x: {target_x} is out of bounds for grid dimensions {self.grid.shape}"
                    raise ValueError(error)
        return distances

    def _localize_obs(self, obs, idx):
        return np.roll(obs, shift=-idx, axis=0)

    def _normalize(self, obs, obs_type):
        # TODO: FIXME
        return obs
        match obs_type:
            case "distance":
                return
            case "positions":
                return
            case "directions":
                return
            case "pos_diff":
                return
            case _:
                raise ValueError(f"Invalid obs_type: {obs_type}")

    def _get_dist(self, player_idx, target_direction):
        d_x, d_y = target_direction
        return

    def step(self, action_dict=None):
        if not action_dict:
            logging.debug("No action dict passed")
            action_dict = {player: 0 for player in self.agents}


        rewards, terminateds = {}, {}
        for i, player in enumerate(self.agents):
            action = action_dict[player]
            reward, terminated = self._step_player(i, action)
            rewards[player] = reward
            terminateds[player] = terminated

        observations = self._get_all_obs()

        # CHECK IF GAME ENDED
        if sum(self.alive) == 1:
            terminateds["__all__"] = True
            winner = f"player_{self.alive.index(1)}"
            rewards[winner] = 1
            logging.debug(f"{winner} wins!")
            self.ended = True
            self.save_replay()
        else:
            terminateds["__all__"] = False

        return observations, rewards, terminateds, {}, {}


    def _step_player(self, player_i, action_num) -> tuple[float, bool]:
        action_name, _, dy_dx = DIR_MAP[action_num]

        logging.debug(f"Player {player_i}: {action_name} ({action_num})")
        self.episode[f"player_{player_i}"].append(action_name)

        old_pos = self.positions[player_i]

        new_y, new_x = old_pos + dy_dx
        if self.grid[new_y, new_x] != 0:
            self.alive[player_i] = 0
            return -1, True

        self.grid[new_y, new_x] = player_i+1
        self.positions[player_i] = [new_y, new_x]

        reward = 0.1 # FIXME
        return reward, False

    def save_replay(self):
        os.makedirs(self.config.log_dir, exist_ok=True)
        unique_id = uuid.uuid4()
        filename = str(unique_id) + ".json"
        filepath = os.path.join(self.config.log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(self.episode, f)


    def sample(self, n=20):
        self.reset()
        render(self.grid)
        for i in range(n):
            self.step()
            render(self.grid)
            if self.ended:
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    env = LightBikeEnv(debug=True)
    env.sample()