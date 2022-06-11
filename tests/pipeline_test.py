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
"""Test for pipeline.py"""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from muzero.pipeline import compute_n_step_target


class PipelineTest(parameterized.TestCase):
    def test_compute_n_step_target_zero_root(self):
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        root_values = [0, 0, 0, 0, 0]
        n_step = 5
        discount = 0.997

        expected_targets = [4.97, 3.982, 2.991, 1.997, 1.0]
        targets = compute_n_step_target(rewards, root_values, n_step, discount)
        np.testing.assert_almost_equal(np.array(targets), np.array(expected_targets), decimal=3)

    def test_compute_n_step_target(self):
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        root_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        n_step = 5
        discount = 0.997

        expected_targets = [
            4.97 + discount**n_step * root_values[0 + n_step],
            4.97 + discount**n_step * root_values[1 + n_step],
            4.97 + discount**n_step * root_values[2 + n_step],
            4.97 + discount**n_step * root_values[3 + n_step],
            4.97 + discount**n_step * root_values[4 + n_step],
            4.97,
            3.982,
            2.991,
            1.997,
            1.0,
        ]
        targets = compute_n_step_target(rewards, root_values, n_step, discount)
        np.testing.assert_almost_equal(np.array(targets), np.array(expected_targets), decimal=3)


if __name__ == '__main__':
    absltest.main()
