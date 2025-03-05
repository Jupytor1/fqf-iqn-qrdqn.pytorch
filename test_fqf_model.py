import os
import yaml
from datetime import datetime
import argparse
from fqf_iqn_qrdqn.env2048 import Env2048
from fqf_iqn_qrdqn.agent import FQFAgent

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', type=str, default=os.path.join('config', 'fqf.yaml'))
parser.add_argument('--model-dir', type=str, default='')
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--render', action='store_true')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

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
agent = FQFAgent(
    env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
    cuda=args.cuda, **config)

agent.load_models(args.model_dir)

agent.online_net.eval()
num_episodes = 0
num_steps = 0
total_return = 0.0

while num_episodes < args.episode:
    state = agent.test_env.reset()
    episode_steps = 0
    episode_return = 0.0
    done = False
    if args.render:
        print(f'========== episode {num_episodes} ==========')
        agent.test_env.print_board()

    while (not done) and episode_steps <= agent.max_episode_steps:
        action = agent.exploit(state)

        next_state, reward, done, _ = agent.test_env.step(action)
        if args.render:
            print(f'action: {action}, reward: {reward}')
            agent.test_env.print_board()
        num_steps += 1
        episode_steps += 1
        episode_return += reward
        state = next_state

    num_episodes += 1
    total_return += episode_return

    print(f'steps: {episode_steps}, return: {episode_return}')

mean_return = total_return / num_episodes
print(f'mean return: {mean_return}')
