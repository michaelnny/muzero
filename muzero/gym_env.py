# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""gym environment processing components."""

# Temporally suppress annoy DeprecationWarning from gym.
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from collections import deque
import gym
from gym import Env
from gym.spaces import Box
import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False)


class NoopResetWrapper(gym.Wrapper):
    def __init__(self, env: Env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class DoneLifeLossWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class NormalizeFrameWrapper(gym.ObservationWrapper):
    """Scale the observation by devide 255.

    Important note:
    This will turn int8 into float32, which will hugely increase memory consumption,
    should also consider state compression when using experience replay.

    """

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation).astype(np.float32) / 255.0


class ResizeGrayscaleFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env: Env, height: int = 96, width: int = 96, grayscale=True):
        """
        Resize frames to 96x96 as done in the MuZero paper, and optionally do gray scale.
        """
        super().__init__(env)
        self._height = height
        self._width = width
        self._grayscale = grayscale

        if self._grayscale:
            num_channels = 1
        else:
            num_channels = 3

        original_space = env.observation_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

        self.observation_space = Box(low=0, high=255, shape=(self._height, self._width, num_channels), dtype=np.int8)

    def observation(self, obs):
        if self._grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        obs = cv2.resize(obs, (self._width, self._height), interpolation=cv2.INTER_AREA)

        if self._grayscale:
            obs = np.expand_dims(obs, -1)

        return obs


class ObservationChannelFirstWrapper(gym.ObservationWrapper):
    """Make observation image channel first, this is for PyTorch only."""

    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        # permute [H, W, C] array to in the range [C, H, W]
        obs = np.array(observation).transpose(2, 0, 1)
        # make sure it's C-contiguous for compress state
        return np.ascontiguousarray(obs, dtype=obs.dtype)


class StackHistoryWrapper(gym.ObservationWrapper):
    """Stack observation history as well as last action as a bias plane."""

    def __init__(self, env: Env, stack_history: int, is_atari: bool):
        super().__init__(env)
        self.stack_history = stack_history
        self.is_atari = is_atari

        if is_atari:
            old_channels = env.observation_space.shape[0]
            # image could be gray scaled or RGB.
            self.old_obs_shape = env.observation_space.shape[1:]
            new_obs_shape = (self.stack_history * (old_channels + 1),) + self.old_obs_shape
        else:
            self.old_obs_shape = env.observation_space.shape
            new_obs_shape = (self.stack_history * 2,) + self.old_obs_shape

        self.num_actions = env.action_space.n

        self.observation_space = Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=new_obs_shape,
            dtype=env.observation_space.dtype,
        )

        self.obs_storage = deque([], maxlen=self.stack_history)

        self.action_storage = deque([], maxlen=self.stack_history)

        self.reset()

    def observation(self):
        stacked_obs = np.stack(list(self.obs_storage), axis=0).astype(np.float32)
        stacked_actions = np.stack(list(self.action_storage), axis=0).astype(np.float32)
        return np.concatenate([stacked_obs, stacked_actions], axis=0)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.obs_storage.appendleft(observation)

        action_plane = self._get_action_bias_plane(action)
        self.action_storage.appendleft(action_plane)

        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        if kwargs.get('return_info', False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused

        for _ in range(self.stack_history):
            self.obs_storage.appendleft(obs)
            dummy_action_plane = self._get_action_bias_plane(0)
            self.action_storage.appendleft(dummy_action_plane)

        if kwargs.get('return_info', False):
            return self.observation(), info
        else:
            return self.observation()

    def _get_action_bias_plane(self, action) -> np.ndarray:
        """Action bias plane is computed as action/num_actions, then broadcast to the shape of observation (unstacked).
        For example in Atari it's a/18."""

        return (action / self.num_actions) * np.ones(self.old_obs_shape).astype(np.float32)


def create_atari_environment(
    env_name: str,
    seed: int = 1,
    stack_history: int = 32,
    screen_height: int = 96,
    screen_width: int = 96,
    noop_max: int = 30,
    max_episode_steps: int = 108000,
    done_on_life_loss: bool = True,
    grayscale: bool = True,
) -> gym.Env:
    """
    Process gym env for Atari games according to the MuZero paper.

    Args:
        env_name: the environment name without 'NoFrameskip' and version.
        seed: seed the runtime.
        stack_history: stack last N frames and the actions lead to those frames.
        screen_height: height of the resized frame.
        screen_width: width of the resized frame.
        noop_max: maximum number of no-ops to apply at the beginning
                of each episode to reduce determinism. These no-ops are applied at a
                low-level, before frame skipping.
        max_episode_steps: maximum steps for an episode.
        done_on_life_loss: if True, mark end of game when loss a life, default on.
        grayscale: if True, gray scale observation image, default on.

    Returns:
        preprocessed gym.Env for Atari games, note the obersevations are not scaled to in the range [0, 1].
    """
    if 'NoFrameskip' in env_name:
        raise ValueError(f'Environment name should not include NoFrameskip, got {env_name}')

    # full_env_name = f'{env_name}NoFrameskip-v4'
    full_env_name = f'{env_name}Deterministic-v4'

    env = gym.make(full_env_name)
    env.reset(seed=seed)

    # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
    # litterature instead of OpenAI Gym's default of 100,000 steps.
    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=max_episode_steps)

    if done_on_life_loss:
        env = DoneLifeLossWrapper(env)

    env = FireResetWrapper(env)
    env = NoopResetWrapper(env, noop_max)
    env = ResizeGrayscaleFrameWrapper(env, screen_height, screen_width, grayscale)
    env = ObservationChannelFirstWrapper(env)
    env = NormalizeFrameWrapper(env)

    if stack_history > 1:
        env = StackHistoryWrapper(env, stack_history, True)

    return env


def create_classic_environment(
    env_name: str,
    seed: int = 1,
    stack_history: int = 1,
) -> gym.Env:
    """
    Process gym env for classic games like CartPole, LunarLander, MountainCar

    Args:
        env_name: the environment name with version attached.
        seed: seed the runtime.

    Returns:
        gym.Env for classic games
    """

    env = gym.make(env_name)
    env.reset(seed=seed)

    if stack_history > 1:
        env = StackHistoryWrapper(env, stack_history, False)

    return env
