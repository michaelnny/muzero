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
from typing import Union

import gym
from gym import Env
from gym.spaces import Box
from gym.error import DependencyNotInstalled
import cv2
import numpy as np


class AtariPreprocessing(gym.Wrapper):
    """Atari 2600 preprocessing wrapper.
    This class follows the guidelines in Machado et al. (2018),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents".
    Specifically, the following preprocess stages applies to the atari environment:
    - Noop Reset: Obtains the initial state by taking a random number of no-ops on reset, default max 30 no-ops.
    - Frame skipping: The number of frames skipped between steps, 4 by default
    - Max-pooling: Pools over the most recent two observations from the frame skips
    - Termination signal when a life is lost: When the agent losses a life during the environment, then the environment is terminated.
        Turned off by default. Not recommended by Machado et al. (2018).
    - Resize to a square image: Resizes the atari environment original observation shape from 210x180 to 84x84 by default
    - Grayscale observation: If the observation is colour or greyscale, by default, greyscale.
    - Scale observation: If to scale the observation between [0, 1) or [0, 255), by default, not scaled.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = False,
        grayscale_obs: bool = True,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
    ):
        """Wrapper for Atari 2600 preprocessing.
        Args:
            env (Env): The environment to apply the preprocessing
            noop_max (int): For No-op reset, the max number no-ops actions are taken at reset, to turn off, set to 0.
            frame_skip (int): The number of frames between new observation the agents observations effecting the frequency at which the agent experiences the game.
            screen_size (int): resize Atari frame
            terminal_on_life_loss (bool): `if True`, then :meth:`step()` returns `done=True` whenever a
                life is lost.
            grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
                is returned.
            grayscale_newaxis (bool): `if True and grayscale_obs=True`, then a channel axis is added to
                grayscale observations to make them 3-dimensional.
            scale_obs (bool): if True, then observation normalized in range [0,1) is returned. It also limits memory
                optimization benefits of FrameStack Wrapper.
        Raises:
            DependencyNotInstalled: opencv-python package not installed
            ValueError: Disable frame-skipping in the original env
        """
        super().__init__(env)
        if cv2 is None:
            raise DependencyNotInstalled(
                "opencv-python package not installed, run `pip install gym[other]` to get dependencies for atari"
            )
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            if "NoFrameskip" not in env.spec.id and getattr(env.unwrapped, "_frameskip", None) != 1:
                raise ValueError(
                    "Disable frame-skipping in the original env. Otherwise, more than one "
                    "frame-skip will happen as through this wrapper"
                )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        if grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]

        self.lives = 0
        self.game_over = False

        _low, _high, _obs_dtype = (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = Box(low=_low, high=_high, shape=_shape, dtype=_obs_dtype)

    def step(self, action):
        """Applies the preprocessing for an :meth:`env.step`."""
        total_reward = 0.0

        for t in range(self.frame_skip):
            _, reward, done, info = self.env.step(action)
            total_reward += reward
            self.game_over = done

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.game_over = done
                self.lives = new_lives

            if done:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB(self.obs_buffer[0])
        return self._get_obs(), total_reward, done, info

    def reset(self, **kwargs):
        """Resets the environment using preprocessing."""
        # NoopReset
        if kwargs.get("return_info", False):
            _, reset_info = self.env.reset(**kwargs)
        else:
            _ = self.env.reset(**kwargs)
            reset_info = {}

        noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            _, _, done, step_info = self.env.step(0)
            reset_info.update(step_info)
            if done:
                if kwargs.get("return_info", False):
                    _, reset_info = self.env.reset(**kwargs)
                else:
                    _ = self.env.reset(**kwargs)
                    reset_info = {}

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)

        if kwargs.get("return_info", False):
            return self._get_obs(), reset_info
        else:
            return self._get_obs()

    @property
    def ale(self):
        """Fix cannot pickle 'ale_py._ale_py.ALEInterface' object error in multiprocessing."""
        return self.env.unwrapped.ale

    def _get_obs(self):
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize(
            self.obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs


# class LazyFrames:
#     """Ensures common frames are only stored once to optimize memory use.
#     To further reduce the memory use, it is optionally to turn on lz4 to compress the observations.
#     Note:
#         This object should only be converted to numpy array just before forward pass.
#     """

#     __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

#     def __init__(self, frames: list, lz4_compress: bool = False):
#         """Lazyframe for a set of frames and if to apply lz4.
#         Args:
#             frames (list): The frames to convert to lazy frames
#             lz4_compress (bool): Use lz4 to compress the frames internally
#         Raises:
#             DependencyNotInstalled: lz4 is not installed
#         """
#         self.frame_shape = tuple(frames[0].shape)
#         self.shape = (len(frames),) + self.frame_shape
#         self.dtype = frames[0].dtype
#         if lz4_compress:
#             try:
#                 from lz4.block import compress
#             except ImportError:
#                 raise DependencyNotInstalled("lz4 is not installed, run `pip install gym[other]`")

#             frames = [compress(frame) for frame in frames]
#         self._frames = frames
#         self.lz4_compress = lz4_compress

#     def __array__(self, dtype=None):
#         """Gets a numpy array of stacked frames with specific dtype.
#         Args:
#             dtype: The dtype of the stacked frames
#         Returns:
#             The array of stacked frames with dtype
#         """
#         arr = self[:]
#         if dtype is not None:
#             return arr.astype(dtype)
#         return arr

#     def __len__(self):
#         """Returns the number of frame stacks.
#         Returns:
#             The number of frame stacks
#         """
#         return self.shape[0]

#     def __getitem__(self, int_or_slice: Union[int, slice]):
#         """Gets the stacked frames for a particular index or slice.
#         Args:
#             int_or_slice: Index or slice to get items for
#         Returns:
#             np.stacked frames for the int or slice
#         """
#         if isinstance(int_or_slice, int):
#             return self._check_decompress(self._frames[int_or_slice])  # single frame
#         return np.stack([self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0)

#     def __eq__(self, other):
#         """Checks that the current frames are equal to the other object."""
#         return self.__array__() == other

#     def _check_decompress(self, frame):
#         if self.lz4_compress:
#             from lz4.block import decompress

#             return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(self.frame_shape)
#         return frame


# class FrameAndActionStack(gym.ObservationWrapper):
#     """Observation wrapper that stacks the observations in a rolling manner.
#     For example, if the number of stacks is 4, then the returned observation contains
#     the most recent 4 observations. For environment 'Pendulum-v1', the original observation
#     is an array with shape [3], so if we stack 4 observations, the processed observation
#     has shape [4, 3].
#     Note:
#         - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
#         - The observation space must be :class:`Box` type. If one uses :class:`Dict`
#           as observation space, it should apply :class:`FlattenObservation` wrapper first.
#           - After :meth:`reset` is called, the frame buffer will be filled with the initial observation. I.e. the observation returned by :meth:`reset` will consist of ``num_stack`-many identical frames,
#     Example:
#         >>> import gym
#         >>> env = gym.make('CarRacing-v1')
#         >>> env = FrameStack(env, 4)
#         >>> env.observation_space
#         Box(4, 96, 96, 3)
#         >>> obs = env.reset()
#         >>> obs.shape
#         (4, 96, 96, 3)
#     """

#     def __init__(self, env: gym.Env, num_stack: int, lz4_compress: bool = False):
#         """Observation wrapper that stacks the observations in a rolling manner.
#         Args:
#             env (Env): The environment to apply the wrapper
#             num_stack (int): The number of frames to stack
#             lz4_compress (bool): Use lz4 to compress the frames internally
#         """
#         super().__init__(env)
#         self.num_stack = num_stack
#         self.lz4_compress = lz4_compress

#         self.frames = deque(maxlen=num_stack)
#         self.actions = deque(maxlen=num_stack)

#         low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
#         high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
#         self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

#     def observation(self, observation):
#         """Converts the wrappers current frames to lazy frames.
#         Args:
#             observation: Ignored
#         Returns:
#             :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
#         """
#         assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
#         stacked_obs = LazyFrames(list(self.frames), self.lz4_compress)
#         stacked_actions = LazyFrames(list(self.actions), self.lz4_compress)

#         return np.concatenate([stacked_obs, stacked_actions], axis=1)

#     def step(self, action):
#         """Steps through the environment, appending the observation to the frame buffer.
#         Args:
#             action: The action to step through the environment with
#         Returns:
#             Stacked observations, reward, done and information from the environment
#         """
#         observation, reward, done, info = self.env.step(action)
#         self.frames.append(observation)
#         self.actions.append(action)
#         return self.observation(None), reward, done, info

#     def reset(self, **kwargs):
#         """Reset the environment with kwargs.
#         Args:
#             **kwargs: The kwargs for the environment reset
#         Returns:
#             The stacked observations
#         """
#         if kwargs.get("return_info", False):
#             obs, info = self.env.reset(**kwargs)
#         else:
#             obs = self.env.reset(**kwargs)
#             info = None  # Unused

#         for _ in range(self.num_stack):
#             self.frames.append(obs)
#             self.actions.append(0)

#         if kwargs.get("return_info", False):
#             return self.observation(None), info
#         else:
#             return self.observation(None)


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

    def __init__(self, env: Env, stack_history: int, is_image: bool):
        super().__init__(env)
        self.stack_history = stack_history
        self.is_image = is_image

        if is_image:
            old_channels = env.observation_space.shape[0]
            # image could be gray scaled or RGB.
            self.old_obs_shape = env.observation_space.shape[1:]
            new_obs_shape = (self.stack_history * (old_channels + 1),) + self.old_obs_shape
        else:
            self.old_obs_shape = env.observation_space.shape
            obs_shape = env.observation_space.shape[0]
            new_obs_shape = (self.stack_history, obs_shape + 1)

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

    def observation(self, observation):
        stacked_obs = np.stack(list(self.obs_storage), axis=0).astype(np.float32)
        stacked_actions = np.stack(list(self.action_storage), axis=0).astype(np.float32)
        if self.is_image:
            # Merge stack and channel dimensions.
            stacked, channel, h, w = stacked_obs.shape
            stacked_obs = stacked_obs.reshape((stacked * channel, h, w))

            return np.concatenate([stacked_obs, stacked_actions], axis=0)
        else:
            return np.concatenate([stacked_obs, stacked_actions], axis=1)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.obs_storage.appendleft(observation)

        action_plane = self._get_action_bias_plane(action)
        self.action_storage.appendleft(action_plane)

        return self.observation(None), reward, done, info

    def reset(self, **kwargs):
        if kwargs.get('return_info', False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused

        for _ in range(self.stack_history):
            self.obs_storage.append(obs)
            dummy_action_plane = self._get_action_bias_plane(0)
            self.action_storage.append(dummy_action_plane)

        if kwargs.get('return_info', False):
            return self.observation(None), info
        else:
            return self.observation(None)

    def _get_action_bias_plane(self, action) -> np.ndarray:
        """Action bias plane is computed as action/num_actions, then broadcast to the shape of observation (unstacked).
        For example in Atari it's a/18."""
        # plus one because action starts from 0
        sacled_action = (action + 1) / self.num_actions
        if self.is_image:
            plane = np.ones(self.old_obs_shape).astype(np.float32)
        else:
            plane = np.ones((1,)).astype(np.float32)

        return sacled_action * plane


class PlayerIdAndActionMaskWrapper(gym.Wrapper):
    """To keep the same configuration with board game, add player ids and action mask to the env."""

    def __init__(self, env: Env):

        gym.Wrapper.__init__(self, env)
        self.current_player = 1
        self.opponent_player = 1
        num_actions = env.action_space.n
        self.actions_mask = np.ones(num_actions, dtype=np.bool8).flatten()


def create_atari_environment(
    env_name: str,
    seed: int = 1,
    stack_history: int = 32,
    frame_skip: int = 4,
    screen_size: int = 96,
    noop_max: int = 30,
    max_episode_steps: int = 108000,
    done_on_life_loss: bool = False,
    grayscale: bool = True,
) -> gym.Env:
    """
    Process gym env for Atari games according to the MuZero paper.

    Args:
        env_name: the environment name.
        seed: seed the runtime.
        stack_history: stack last N frames and the actions lead to those frames.
        frame_skip: skip last N frames by repeat action.
        screen_size: width and height of the resized frame.
        noop_max: maximum number of no-ops to apply at the beginning
                of each episode to reduce determinism. These no-ops are applied at a
                low-level, before frame skipping.
        max_episode_steps: maximum steps for an episode.
        done_on_life_loss: if True, mark end of game when loss a life, default on.
        grayscale: if True, gray scale observation image, default on.

    Returns:
        preprocessed gym.Env for Atari games, note the obersevations are not scaled to in the range [0, 1].
    """

    env = gym.make(env_name)
    env.reset(seed=seed)

    # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
    # litterature instead of OpenAI Gym's default of 100,000 steps.
    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=max_episode_steps)

    env = AtariPreprocessing(env, noop_max, frame_skip, screen_size, done_on_life_loss, grayscale, True, True)
    env = ObservationChannelFirstWrapper(env)

    if stack_history > 1:
        env = StackHistoryWrapper(env, stack_history, True)

    env = PlayerIdAndActionMaskWrapper(env)
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

    env = PlayerIdAndActionMaskWrapper(env)
    return env
