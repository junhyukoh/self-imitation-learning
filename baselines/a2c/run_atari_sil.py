#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c_sil import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, sil_update, sil_beta):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    env_args = {'episode_life': False, 'clip_rewards': False}
    env = VecFrameStack(
            make_atari_env(env_id, num_env, seed, wrapper_kwargs=env_args), 4)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, 
          sil_update=sil_update, sil_beta=sil_beta)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--sil-update', type=int, default=4, help="Number of updates per iteration")
    parser.add_argument('--sil-beta', type=float, default=0.1, help="Beta for weighted IS")
    parser.add_argument('--log', default='/tmp/a2c')
    args = parser.parse_args()
    logger.configure(dir=args.log)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, 
        sil_update=args.sil_update, sil_beta=args.sil_beta,
        num_env=16)

if __name__ == '__main__':
    main()
