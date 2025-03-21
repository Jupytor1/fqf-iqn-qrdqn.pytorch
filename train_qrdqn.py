import os
import yaml
import argparse
from datetime import datetime

# from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.env2048 import Env2048
from fqf_iqn_qrdqn.agent import QRDQNAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments.
    # env = make_pytorch_env(args.env_id)
    # test_env = make_pytorch_env(
    #     args.env_id, episode_life=False, clip_rewards=False)
    env = Env2048()
    test_env = Env2048()

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    # log_dir = os.path.join(
    #     'logs', args.env_id, f'{name}-seed{args.seed}-{time}')
    log_dir = os.path.join(
        'logs', f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    agent = QRDQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    if args.model_dir is not None:
        agent.load_models(args.model_dir)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'qrdqn.yaml'))
    # parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default=None)
    args = parser.parse_args()
    run(args)
