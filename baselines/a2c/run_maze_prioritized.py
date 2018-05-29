#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_maze_env, maze_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c_prioritized_sil_maze import learn
from baselines.a2c.policies import MlpPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, sil_update, sil_beta, count_exp_weight):
    if policy == 'mlp':
        policy_fn = MlpPolicy
    env = VecFrameStack(make_maze_env(env_id, num_env, seed), 4)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule,
          sil_update=sil_update, sil_beta=sil_beta, count_exp_weight=count_exp_weight, 
          gamma=0.95, ent_coef=0.03)
    env.close()

def main():
    parser = maze_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['mlp'], default='mlp')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--sil-update', type=int, default=4, help="Number of updates per iteration")
    parser.add_argument('--sil-beta', type=float, default=0.4, help="Beta for weighted IS")
    parser.add_argument('--count-exp-weight', type=float, default=0, help='weight of count exploration')
    parser.add_argument('--log', default='/tmp/a2c')
    args = parser.parse_args()
    logger.configure(dir=args.log)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, 
        sil_update=args.sil_update, sil_beta=args.sil_beta, 
        count_exp_weight=args.count_exp_weight, num_env=16)

if __name__ == '__main__':
    main()
