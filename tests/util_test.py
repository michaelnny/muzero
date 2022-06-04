"""Test for util.py"""
from absl.testing import absltest
from absl.testing import parameterized
import torch
import numpy as np

from muzero.util import transform_to_2hot


class PipelineTest(parameterized.TestCase):
    def test_transform_to_2hot(self):

        x = torch.tensor(
            [
                [3.7],
                [2.3],
            ],
            dtype=torch.float32,
        )

        expected = np.array(
            [
                [
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3000, 0.7000, 0.0000],
                ],
                [
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7000, 0.3000, 0.0000, 0.0000],
                ],
            ],
            dtype=np.float32,
        )
        probs = transform_to_2hot(x, -5, 5, 11)
        np.testing.assert_almost_equal(probs.numpy(), expected, decimal=4)


if __name__ == '__main__':
    absltest.main()
