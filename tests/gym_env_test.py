# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for gym_env.py."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import gym
from muzero import gym_env


class BumpUpReward(gym.RewardWrapper):
    'Bump up rewards so later can use it to test clip reward'

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        noise = np.random.randint(10, 100)
        if np.random.rand() < 0.5:
            return reward + noise
        return reward - noise


class AtariEnvironmentTest(parameterized.TestCase):
    @parameterized.named_parameters(('environment_pong', 'Pong'), ('environment_breakout', 'Breakout'))
    def test_run_step(self, environment_name):
        seed = 1
        env = gym_env.create_atari_environment(env_name=environment_name, seed=seed)
        env.reset()
        env.step(0)
        env.close()

    def test_environment_name_exception(self):
        environment_name = 'PongNoFrameskip-v4'
        seed = 1

        with self.assertRaisesRegex(ValueError, 'Environment name should not include NoFrameskip, got PongNoFrameskip-v4'):
            env = gym_env.create_atari_environment(env_name=environment_name, seed=seed)
            env.reset()
            env.step(0)
            env.close()

    @parameterized.named_parameters(
        ('sizes_84x84x1', (84, 84, 1)), ('sizes_84x84x4', (84, 84, 4)), ('sizes_96x72x8', (96, 72, 8))
    )
    def test_env_channel_last_different_sizes(self, sizes):
        seed = 1

        env = gym_env.create_atari_environment(
            env_name='Pong',
            seed=seed,
            screen_height=sizes[0],
            screen_width=sizes[1],
            frame_stack=sizes[2],
            channel_first=False,
            scale_obs=False,
        )

        if sizes[2] > 1:
            expected_obs_shape = (sizes[0], sizes[1], sizes[2] * 2)
            expected_dtype = np.float32
        else:
            expected_obs_shape = sizes
            expected_dtype = np.uint8

        obs = env.reset()

        self.assertEqual(env.observation_space.shape, expected_obs_shape)
        self.assertEqual(env.observation_space.dtype, expected_dtype)
        self.assertEqual(obs.shape, expected_obs_shape)
        self.assertEqual(obs.dtype, expected_dtype)
        # self.assertTrue(obs.flags['C_CONTIGUOUS'])

        for _ in range(3):  # 3 games
            obs = env.reset()
            for _ in range(20):  # each game 20 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                self.assertEqual(obs.shape, expected_obs_shape)
                self.assertEqual(obs.dtype, expected_dtype)
                if done:
                    break
        env.close()

    @parameterized.named_parameters(
        ('sizes_1x84x84', (1, 84, 84)), ('sizes_4x84x84', (4, 84, 84)), ('sizes_8x96x72', (8, 96, 72))
    )
    def test_env_channel_first_different_sizes(self, sizes):
        seed = 1

        env = gym_env.create_atari_environment(
            env_name='Pong',
            seed=seed,
            screen_height=sizes[1],
            screen_width=sizes[2],
            frame_stack=sizes[0],
            channel_first=True,
            scale_obs=False,
        )

        if sizes[0] > 1:
            expected_obs_shape = (sizes[0] * 2, sizes[1], sizes[2])
            expected_dtype = np.float32
        else:
            expected_obs_shape = (sizes[0], sizes[1], sizes[2])
            expected_dtype = np.uint8

        obs = env.reset()
        expected_dtype = np.float32
        self.assertEqual(env.observation_space.shape, expected_obs_shape)
        self.assertEqual(env.observation_space.dtype, expected_dtype)
        self.assertEqual(obs.shape, expected_obs_shape)
        self.assertEqual(obs.dtype, expected_dtype)
        # self.assertTrue(obs.flags['C_CONTIGUOUS'])

        for _ in range(3):  # 3 games
            obs = env.reset()
            for _ in range(20):  # each game 20 steps
                obs, r, done, _ = env.step(env.action_space.sample())
                self.assertEqual(obs.shape, expected_obs_shape)
                self.assertEqual(obs.dtype, expected_dtype)
                if done:
                    break
        env.close()


if __name__ == '__main__':
    absltest.main()
