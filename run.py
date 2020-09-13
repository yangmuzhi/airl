#!/usr/bin/python3
from network_models import d_net, Policy_net
from algo import Discriminator, PPO, AIRL_wrapper
import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='log directory', default='log/')
parser.add_argument('--savedir', help='save directory', default='trained_models/')
parser.add_argument('--gamma', default=0.95, type=float)
parser.add_argument('--iters', default=int(1e4), type=int)
args = parser.parse_args()

obs_dims = (4,)
n_actions = 2

agent = PPO(args.savedir, Policy_net, (4,), 2)
D = Discriminator(args.savedir, obs_dims, n_actions)
env = gym.make("CartPole-v0")
trainer = AIRL_wrapper(agent, D, env, args.savedir)

trainer.train(args.iters)