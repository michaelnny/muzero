#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run smoke tests
python3 -m muzero.tictactoe.run_training
python3 -m muzero.classic.run_training
python3 -m muzero.atari.run_training_ram
