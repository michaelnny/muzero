from absl import app
from absl import flags
from absl import logging
import os
import gym
import torch

from muzero.network import MuZeroMLPNet
from muzero.gym_env import create_classic_environment
from muzero.pipeline import load_checkpoint
from muzero.mcts import uct_search

FLAGS = flags.FLAGS
flags.DEFINE_string("environment_name", 'LunarLander-v2', "Classic problem like 'CartPole-v1', 'LunarLander-v2'")
flags.DEFINE_integer("stack_history", 4, "Stack previous states.")

flags.DEFINE_integer('num_planes', 256, 'Number of hidden units for the FC layers in the model.')
flags.DEFINE_integer('value_support_size', 31, 'Value scalar projection support size, [-15, 15].')
flags.DEFINE_integer('reward_support_size', 31, 'Reward scalar projection support size, [-15, 15].')
flags.DEFINE_integer('hidden_size', 64, 'Hidden state vector size.')

flags.DEFINE_float('discount', 0.997, 'Gamma discount.')
flags.DEFINE_integer('num_simulations', 30, 'Number of simulations per MCTS search, per agent environment time step.')
flags.DEFINE_float(
    'root_noise_alpha', 0.0, 'Alph for dirichlet noise to MCTS root node prior probabilities to encourage exploration.'
)
flags.DEFINE_integer('seed', 1, 'Seed the runtime.')

flags.DEFINE_string(
    'load_checkpoint_file',
    'checkpoints/classic/LunarLander-v2_train_steps_268000',
    'Load the checkpoint from file.',
)
flags.DEFINE_string(
    'record_video_dir',
    'recording/classic',
    'Record play video.',
)


def main(argv):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    runtime_device = torch.device(device)

    eval_env = create_classic_environment(FLAGS.environment_name, FLAGS.seed, FLAGS.stack_history)
    input_shape = eval_env.observation_space.shape
    num_actions = eval_env.action_space.n

    network = MuZeroMLPNet(
        input_shape, num_actions, FLAGS.num_planes, FLAGS.value_support_size, FLAGS.reward_support_size, FLAGS.hidden_size
    )

    # Load states from checkpoint to resume training.
    if FLAGS.load_checkpoint_file is not None and os.path.isfile(FLAGS.load_checkpoint_file):
        loaded_state = load_checkpoint(FLAGS.load_checkpoint_file, 'cpu')
        network.load_state_dict(loaded_state['network'])
        logging.info(f'Loaded state from checkpoint {FLAGS.load_checkpoint_file}')

    network.eval()

    if FLAGS.record_video_dir is not None and os.path.isdir(FLAGS.record_video_dir):
        eval_env = gym.wrappers.RecordVideo(eval_env, FLAGS.record_video_dir)

    steps = 0
    returns = 0.0

    obs = eval_env.reset()
    while True:
        action, *_ = uct_search(
            state=obs,
            network=network,
            device=runtime_device,
            discount=FLAGS.discount,
            temperature=0.1,
            num_simulations=FLAGS.num_simulations,
            root_noise_alpha=FLAGS.root_noise_alpha,
            actions_mask=eval_env.actions_mask,
            current_player=eval_env.current_player,
            opponent_player=eval_env.opponent_player,
            competitive=False,
            best_action=True,
        )

        obs, reward, done, _ = eval_env.step(action)
        steps += 1
        returns += reward

        if done:
            break

    eval_env.close()
    logging.info(f'Episode returns: {returns}, steps: {steps}')


if __name__ == '__main__':
    app.run(main)
