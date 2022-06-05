# MuZero
## A PyTorch implementation of DeepMind's MuZero agent


## Environment and requirements
* Python        3.9.12
* pip           22.0.3
* PyTorch       1.11.0
* gym           0.23.1
* numpy         1.21.6


## Code structure

* each directory `atari`, `classic`, `gomoku`, `tictactoe` contains the following modules:
  - `run_training.py` trains the agent for a specific game/control problem.
  - `eval_agent.py` evaluate the trained agent by loading from checkpoint.
* `core.py` contains the MuZero configuration for different game/control problem.
* `games` directory contains the custom Gomoku and Tic-Tac-Toe board game env implemented with openAI Gym.
* `mcts.py` contains the MCTS node and UCT tree-search algorithm.
* `pipeline.py` contains the functions to run self-play, training, and evaluation loops.
* `trackers.py` contains the functions to monitoring traning progress using Tensorboard.


## Note
* Only tested on classic control problems and Tic-Tac-Toe
* It does not converge on LunarLander.


## Quick start
### Install required packages
```
# install homebrew, skip this step if already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# upgrade pip
python3 -m pip install --upgrade pip setuptools


pip3 install -r requirements.txt
```


### Start training

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


## Monitoring performance and statistics with Tensorboard

```
tensorboard --logdir=runs
```


### Evaluate agents
Note for board games with two players, the evaluation will run in `MuZero vs. MuZero` mode.

To start play the game, make sure you have a valid checkpoint file and run the following command
```
python3 -m muzero.tictactoe.eval_agent

python3 -m muzero.classic.eval_agent

```


### Screenshots Tic-Tac-Toe
* Training performance measured in elo rating
![Training performance](../main/screenshots/tictactoe.png)


### Screenshots CartPole
![Training performance](../main/screenshots/cartpole.png)


### Screenshots LunarLander
![Training performance](../main/screenshots/lunarlander.png)


## Reference Papers
* [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265)


## Reference Code
* [Model based RL](https://github.com/JimOhman/model-based-rl)
* [Alpha Zero](https://github.com/michaelnny/alpha_zero)
* [Deep RL Zoo](https://github.com/michaelnny/deep_rl_zoo)


## Citing our work

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
