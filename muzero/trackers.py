import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

from pathlib import Path
import shutil
import timeit
from typing import Mapping, Text
import numpy as np

from torch.utils.tensorboard import SummaryWriter


class TensorboardEpisodTracker:
    """TensorboardEpisodTracker to write to tensorboard"""

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
        self._current_episode_rewards.append(reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if done:
            self._episode_returns.append(sum(self._current_episode_rewards))
            self._episode_steps.append(self._current_episode_step)
            self._current_episode_rewards = []
            self._current_episode_step = 0

            # To improve performance, only logging at end of an episode.
            tb_steps = self._num_steps_since_reset
            num_episodes = len(self._episode_returns)

            # tracker per episode
            episode_return = self._episode_returns[-1]
            episode_step = self._episode_steps[-1]

            # tracker per step
            self._writer.add_scalar('self_play(steps)/num_episodes', num_episodes, tb_steps)
            self._writer.add_scalar('self_play(steps)/episode_return', episode_return, tb_steps)
            self._writer.add_scalar('self_play(steps)/episode_steps', episode_step, tb_steps)

            time_stats = self.get_time_stat()
            self._writer.add_scalar('self_play(steps)/run_duration(minutes)', time_stats['duration'] / 60, tb_steps)
            self._writer.add_scalar('self_play(steps)/step_rate(second)', time_stats['step_rate'], tb_steps)

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


class TensorboardLearnerTracker:
    """TensorboardLearnerTracker to write to tensorboard"""

    # def __init__(self, log_dir: str):
    def __init__(self, writer: SummaryWriter):
        self._num_steps_since_reset = None
        self._start = None
        self._writer = writer

    def reset(self) -> None:
        """Resets all gathered statistics, not to be called between episodes."""
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()

    def step(self, loss, lr) -> None:
        self._num_steps_since_reset += 1
        tb_steps = self._num_steps_since_reset

        # tracker per step
        self._writer.add_scalar('learner(train_steps)/loss', loss, tb_steps)
        self._writer.add_scalar('learner(train_steps)/learning_rate', lr, tb_steps)

        time_stats = self.get_time_stat()
        self._writer.add_scalar('learner(train_steps)/step_rate(minutes)', time_stats['step_rate'] / 60, tb_steps)

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


def make_self_play_trackers(run_log_dir='actor'):
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
        TensorboardEpisodTracker(writer),
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
        TensorboardLearnerTracker(writer),
    ]
