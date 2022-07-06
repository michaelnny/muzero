MuZero
=============================
A PyTorch implementation of DeepMind's MuZero agent.


# Content
- [Environment and Requirements](#environment-and-requirements)
- [Code Structure](#code-structure)
- [Author's Notes](#authors-notes)
- [Quick Start](#quick-start)
- [Start Training](#start-training)
- [Monitoring with Tensorboard](#monitoring-with-tensorboard)
- [Evaluate Agents](#evaluate-agents)
- [Reference Papers](#reference-papers)
- [Reference Code](#reference-code)
- [License](#license)
- [Citing our work](#citing-our-work)

# Environment and Requirements
* Python        3.9.12
* pip           22.0.3
* PyTorch       1.11.0
* gym           0.23.1
* numpy         1.21.6


# Code Structure

* each of the (`atari`, `classic`, `gomoku`, `tictactoe`) directory contains the following modules:
  - `run_training.py` trains the agent for a specific game/control problem.
  - `eval_agent.py` evaluate the trained agent by loading from checkpoint.
* `config.py` contains the MuZero configuration for different game/control problem.
* `games` directory contains the custom Gomoku and Tic-Tac-Toe board game env implemented with openAI Gym.
* `gym_env.py` contains openAI Gym wrappers for both Atari and classic control problem.
* `mcts.py` contains the MCTS node and UCT tree-search algorithm.
* `pipeline.py` contains the functions to run self-play, training, and evaluation loops.
* `util.py` contains the functions for value and reward target transform and rescaling.
* `replay.py` contains the experience replay class.
* `trackers.py` contains the functions to monitoring training progress using Tensorboard.


# Author's Notes
* Only tested on classic control tasks and Tic-Tac-Toe.
* Hyper-parameters are not fine-tuned.
* We use uniform random replay as it seems to be better than prioritized replay.


# Quick Start
## Install required packages on Mac
```
# install homebrew, skip this step if already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# upgrade pip
python3 -m pip install --upgrade pip setuptools

# install ffmpeg for recording agent self-play
brew install ffmpeg

# install snappy for compress numpy.array on M1 mac
brew install snappy
CPPFLAGS="-I/opt/homebrew/include -L/opt/homebrew/lib" pip3 install python-snappy

pip3 install -r requirements.txt
```


## Install required packages on Ubuntu Linux
```
# install swig which is required for box-2d
sudo apt install swig

# install ffmpeg for recording agent self-play
sudo apt-get install ffmpeg

# upgrade pip
python3 -m pip install --upgrade pip setuptools

pip3 install -r requirements.txt
```


# Start Training

```
# Training on classic control problems
python3 -m muzero.classic.run_training
python3 -m muzero.classic.run_training --environment_name=LunarLander-v2

# Training on Atari
python3 -m muzero.atari.run_training

# Training on Tic-Tac-Toe
python3 -m muzero.tictactoe.run_training

# Training on Gomoku
python3 -m muzero.gomoku.run_training
```


# Monitoring with Tensorboard

```
tensorboard --logdir=runs
```

## Screenshots Tic-Tac-Toe
* Training performance measured in Elo rating
![Training performance](/screenshots/TicTacToe.png)


## Screenshots CartPole
![Training performance](/screenshots/CartPole.png)


## Screenshots LunarLander
![Training performance](/screenshots/LunarLander.png)


# Evaluate Agents
Note for board games with two players, the evaluation will run in `MuZero vs. MuZero` mode.

To start play the game, make sure you have a valid checkpoint file and run the following command
```
python3 -m muzero.tictactoe.eval_agent

python3 -m muzero.classic.eval_agent

```


# Reference Papers
* [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)


# Reference Code
* [Model based RL](https://github.com/JimOhman/model-based-rl)
* [Alpha Zero](https://github.com/michaelnny/alpha_zero)
* [Deep RL Zoo](https://github.com/michaelnny/deep_rl_zoo)


# License

This project is licensed under the Apache License, Version 2.0 (the "License")
see the LICENSE file for details


# Citing our work

If you reference or use our project in your research, please cite:

```
@software{muzero2022github,
  title = {{MuZero}: A PyTorch implementation of DeepMind's MuZero agent},
  author = {Michael Hu},
  url = {https://github.com/michaelnny/muzero},
  version = {1.0.0},
  year = {2022},
}
```
