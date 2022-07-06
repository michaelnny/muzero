#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run unit tests
python3 -m tests.classic.run_training_test
python3 -m tests.atari.run_training_test
python3 -m tests.tictactoe.run_training_test
python3 -m tests.gomoku.run_training_test
