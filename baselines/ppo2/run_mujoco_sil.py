#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger


def train(env_id, num_timesteps, seed, lr,
        sil_update, sil_value, sil_alpha, sil_beta):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2_sil
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env():
        env = gym.make(env_id)
        env = bench.Monitor(env, logger.get_dir(), allow_early_resets=True)
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = ppo2_sil.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
                       lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
                       ent_coef=0.0,
                       lr=lr,
                       cliprange=0.2,
                       total_timesteps=num_timesteps,
                       sil_update=sil_update,
                       sil_value=sil_value,
                       sil_alpha=sil_alpha,
                       sil_beta=sil_beta,
                       )

    return model, env


def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--sil-update', type=float, default=10, help="Number of updates per iteration")
    parser.add_argument('--sil-value', type=float, default=0.01, help="Weight for value update")
    parser.add_argument('--sil-alpha', type=float, default=0.6, help="Alpha for prioritized replay")
    parser.add_argument('--sil-beta', type=float, default=0.1, help="Beta for prioritized replay")

    args = parser.parse_args()
    logger.configure()
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            lr=args.lr,
            sil_update=args.sil_update, sil_value=args.sil_value,
            sil_alpha=args.sil_alpha, sil_beta=args.sil_beta)

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:]  = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    main()
