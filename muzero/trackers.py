# Copyright 2022 Michael Hu. All Rights Reserved.
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
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from pathlib import Path
import shutil
import timeit
from typing import Mapping, Text, List
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class ActorTracker:
    """Actor tracker to write to tensorboard"""

    def __init__(self, writer: SummaryWriter):
        self._num_steps_since_reset = None
        self._episode_returns = None
        self._episode_steps = None
        self._current_episode_rewards = None
        self._current_episode_step = None

        self._start = None
        self._writer = writer

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._episode_returns = []
        self._episode_steps = []
        self._current_episode_step = 0
        self._current_episode_rewards = []

        self._start = timeit.default_timer()

    def step(self, reward, done) -> None:
        """Accumulate episode rewards and steps statistics."""
        self._current_episode_rewards.append(reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if done:
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._episode_steps.append(self._current_episode_step)
            self._current_episode_rewards = []
            self._current_episode_root_values = []
            self._current_episode_step = 0

            # To improve performance, only logging at end of an episode.
            tb_steps = self._num_steps_since_reset
            num_episodes = len(self._episode_returns)

            # tracker per episode
            episode_return = self._episode_returns[-1]
            episode_step = self._episode_steps[-1]

            # tracker per step
            self._writer.add_scalar('actor(env_steps)/num_episodes', num_episodes, tb_steps)
            self._writer.add_scalar('actor(env_steps)/episode_return', episode_return, tb_steps)
            self._writer.add_scalar('actor(env_steps)/episode_steps', episode_step, tb_steps)

            time_stats = self.get_time_stat()
            # self._writer.add_scalar('actor(env_steps)/run_duration(minutes)', time_stats['duration'] / 60, tb_steps)
            self._writer.add_scalar('actor(env_steps)/step_rate(second)', time_stats['step_rate'], tb_steps)

    def get_time_stat(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps': self._num_steps_since_reset,
            'duration': duration,
        }


class LearnerTracker:
    """Learner tracker to write to tensorboard"""

    def __init__(self, writer: SummaryWriter):
        self._num_steps_since_reset = None
        self._start = None
        self._writer = writer

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def step(self, loss, lr, train_steps) -> None:
        """Log training loss statistics."""
        self._num_steps_since_reset = train_steps

        self._writer.add_scalar('learner(train_steps)/loss', loss, train_steps)
        self._writer.add_scalar('learner(train_steps)/learning_rate', lr, train_steps)

        time_stats = self.get_time_stat()
        self._writer.add_scalar('learner(train_steps)/step_rate(minutes)', time_stats['step_rate'] * 60, train_steps)
        self._writer.add_scalar('learner(train_steps)/run_duration(minutes)', time_stats['duration'] / 60, train_steps)

    def get_time_stat(self) -> Mapping[Text, float]:
        """Returns statistics as a dictionary."""
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps': self._num_steps_since_reset,
            'duration': duration,
        }


class EvaluatorTracker:
    """Evaluator tracker to write to tensorboard"""

    def __init__(self, writer: SummaryWriter):
        self._num_steps_since_reset = None
        self._writer = writer

        # # Use custom layout.
        # layout = {
        #     "evaluator(train_steps)": {
        #         "episode_returns": ["Multiline", ["episode_returns/min", "episode_returns/max", "episode_returns/mean"]],
        #         "episode_steps": ["Multiline", ["episode_steps/min", "episode_steps/max", "episode_steps/mean"]],
        #     },
        # }

        # self._writer.add_custom_scalars(layout)

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0

    def step(self, episode_returns: List[int], episode_steps: List[int], train_steps: int) -> None:
        self._num_steps_since_reset = train_steps

        # self._writer.add_scalar('episode_returns/min', np.min(episode_returns), self._num_steps_since_reset)
        # self._writer.add_scalar('episode_returns/max', np.max(episode_returns), self._num_steps_since_reset)
        # self._writer.add_scalar('episode_returns/mean', np.mean(episode_returns), self._num_steps_since_reset)
        # self._writer.add_scalar('episode_steps/min', np.min(episode_steps), self._num_steps_since_reset)
        # self._writer.add_scalar('episode_steps/max', np.max(episode_steps), self._num_steps_since_reset)
        # self._writer.add_scalar('episode_steps/mean', np.mean(episode_steps), self._num_steps_since_reset)

        self._writer.add_scalar(
            'evaluator(train_steps)/mean_episode_returns', np.mean(episode_returns), self._num_steps_since_reset
        )
        self._writer.add_scalar(
            'evaluator(train_steps)/mean_episode_steps', np.mean(episode_steps), self._num_steps_since_reset
        )


class BoardGameEvaluatorTracker:
    """Evaluator tracker for Board Game to write to tensorboard"""

    def __init__(self, writer: SummaryWriter):
        self._num_steps_since_reset = None
        self._writer = writer

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0

    def step(self, elo, episode_steps, train_steps) -> None:
        self._num_steps_since_reset = train_steps

        # tracker per checkpoint
        self._writer.add_scalar('evaluator(train_steps)/elo_rating', elo, self._num_steps_since_reset)
        self._writer.add_scalar('evaluator(train_steps)/episode_steps', episode_steps, self._num_steps_since_reset)


def make_actor_trackers(run_log_dir='actor'):
    """
    Create trackers for the training/evaluation run.

    Args:
        run_log_dir: tensorboard run log directory.
    """

    tb_log_dir = Path(f'runs/{run_log_dir}')

    # Remove existing log directory
    if tb_log_dir.exists() and tb_log_dir.is_dir():
        shutil.rmtree(tb_log_dir)

    writer = SummaryWriter(tb_log_dir)

    return [
        ActorTracker(writer),
    ]


def make_learner_trackers(run_log_dir='learner'):
    """
    Create trackers for the training/evaluation run.

    Args:
        run_log_dir: tensorboard run log directory.
    """

    tb_log_dir = Path(f'runs/{run_log_dir}')

    # Remove existing log directory
    if tb_log_dir.exists() and tb_log_dir.is_dir():
        shutil.rmtree(tb_log_dir)

    writer = SummaryWriter(tb_log_dir)

    return [
        LearnerTracker(writer),
    ]


def make_evaluator_trackers(run_log_dir='evaluator', is_board_game: bool = False):
    """
    Create trackers for the evaluator run.

    Args:
        run_log_dir: tensorboard run log directory.
    """

    tb_log_dir = Path(f'runs/{run_log_dir}')

    # Remove existing log directory
    if tb_log_dir.exists() and tb_log_dir.is_dir():
        shutil.rmtree(tb_log_dir)

    writer = SummaryWriter(tb_log_dir)

    if is_board_game:
        return [BoardGameEvaluatorTracker(writer)]

    return [EvaluatorTracker(writer)]
