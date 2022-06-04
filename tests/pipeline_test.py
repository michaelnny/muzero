"""Test for pipeline.py"""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from muzero.pipeline import compute_n_step_target


class PipelineTest(parameterized.TestCase):
    def test_compute_n_step_target(self):
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        root_values = [0, 0, 0, 0, 0]
        n_step = 5
        discount = 0.997

        expected_targets = [4.97, 3.982, 2.991, 1.997, 1.0]
        targets = compute_n_step_target(rewards, root_values, n_step, discount)
        np.testing.assert_almost_equal(np.array(targets), np.array(expected_targets), decimal=3)


if __name__ == '__main__':
    absltest.main()
