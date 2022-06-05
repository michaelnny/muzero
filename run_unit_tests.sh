#!/bin/bash

set -u -e  # Check for uninitialized variables and exit if any command fails.

# Run unit tests
python3 -m tests.games.boardgame_test
python3 -m tests.games.gomoku_test
python3 -m tests.games.tictactoe_test
python3 -m tests.util_test
python3 -m tests.pipeline_test
