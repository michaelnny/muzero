"""Tests for gomoku.run_training."""

from pathlib import Path
import shutil
from absl import flags
from absl.testing import flagsaver
from absl.testing import absltest
from muzero.gomoku import run_training

FLAGS = flags.FLAGS
FLAGS.checkpoint_dir = '/tmp/e2e_test_checkpoint'
FLAGS.use_tensorboard = False
FLAGS.num_actors = 2
FLAGS.replay_capacity = 100
FLAGS.min_replay_size = 10
FLAGS.num_training_steps = 10


class RunGomokuGameTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.checkpoint_dir = Path(FLAGS.checkpoint_dir)

    @flagsaver.flagsaver
    def test_can_run_agent(self):
        FLAGS.batch_size = 8
        FLAGS.clip_grad = True
        run_training.main(None)

    def tearDown(self) -> None:
        # Clean up
        try:
            shutil.rmtree(self.checkpoint_dir)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    absltest.main()
