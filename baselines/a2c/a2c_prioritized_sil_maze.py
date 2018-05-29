import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.prioritized_self_imitation import SelfImitation
from baselines.common.runners import AbstractEnvRunner
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse
from baselines.a2c.utils import EpisodeStats
from baselines.a2c.utils import save_heatmap_from_array

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
            sil_update=4, sil_beta=0.0):

        sess = tf_util.make_session()
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)
        sil_model = policy(sess, ob_space, ac_space, nenvs, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef
        value_avg = tf.reduce_mean(train_model.vf)

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, v_avg, _ = sess.run(
                [pg_loss, vf_loss, entropy, value_avg, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, v_avg

        self.sil = SelfImitation(sil_model.X, sil_model.vf, 
                sil_model.entropy, sil_model.value, sil_model.neg_log_prob,
                ac_space, np.copy, n_env=nenvs, n_update=sil_update, beta=sil_beta, stack=1)
        self.sil.build_train_op(params, trainer, LR, max_grad_norm=max_grad_norm)
        
        def sil_train():
            cur_lr = lr.value()
            return self.sil.train(sess, cur_lr)

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.sil_train = sil_train
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99, count_exp_weight=0.0):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.count_exp_weight = count_exp_weight
        self.count = {}

    def ob_to_str(self, ob):
        total_str = ''
        for i in range(ob.size):
            total_str += str(int(ob[i])) + ','
        return total_str

    def count_state(self, ob):
        s = self.ob_to_str(ob)
        if s not in self.count.keys():
            self.count[s] = 0
        self.count[s] += 1
        # print(ob, s, self.count[s])
        return self.count[s]

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_raw_rewards = [],[],[],[],[],[]
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, raw_rewards, dones, _ = self.env.step(actions)
            rewards = np.zeros(len(raw_rewards))
            for i in range(len(rewards)):
                if self.count_exp_weight > 0:
                    count = self.count_state(obs[i])
                    rewards[i] = raw_rewards[i]+self.count_exp_weight/np.sqrt(count)
                else:
                    rewards[i] = raw_rewards[i]
            self.states = states
            self.dones = dones
            if hasattr(self.model, 'sil'):
                self.model.sil.step(self.obs, actions, raw_rewards, dones)
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
            mb_raw_rewards.append(raw_rewards)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_raw_rewards = np.asarray(mb_raw_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_raw_rewards = mb_raw_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_raw_rewards

    def show_heatmap(self):
        maze_size = 13
        without_key_room = np.zeros([maze_size, maze_size])
        with_key_room = np.zeros([maze_size, maze_size])
        room = np.zeros([maze_size, maze_size])
        for state_str in self.count.items():
            state_number = state_str.split(',')
            x = int(state_number[1]) 
            y = int(state_number[2])
            key = int(state_number[3])   
            room[y][x] += 1
            if key > 0:
                with_key_room[y][x] += 1
            else:
                without_key_room[y][x] += 1
        save_heatmap_from_array(without_key_room, logger.get_dir()+'/without_key_room')
        save_heatmap_from_array(with_key_room, logger.get_dir()+'/with_key_room')
        save_heatmap_from_array(room, logger.get_dir()+'/room')


def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100, sil_update=4, sil_beta=0.0, count_exp_weight=0.0):
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, sil_update=sil_update, sil_beta=sil_beta)
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma, count_exp_weight=count_exp_weight)

    episode_stats = EpisodeStats(nsteps, nenvs)
    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, raw_rewards = runner.run()
        episode_stats.feed(raw_rewards, masks)
        policy_loss, value_loss, policy_entropy, v_avg = model.train(obs, states, rewards, masks, actions, values)
        sil_loss, sil_adv, sil_samples, sil_nlogp = model.sil_train()
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("episode_raw_reward", episode_stats.mean_reward())
            logger.record_tabular("good_trajectory_raw_reward", float(model.sil.get_best_reward()))
            logger.record_tabular("sil_num_episodes", float(model.sil.num_episodes()))
            if sil_update > 0:
                logger.record_tabular("sil_loss", float(sil_loss))
                logger.record_tabular("sil_adv", float(sil_adv))
                logger.record_tabular("sil_valid_samples", float(sil_samples))
                logger.record_tabular("sil_steps", float(model.sil.num_steps()))
                logger.record_tabular("sil_mean_neg_logp", np.mean(sil_nlogp))
                logger.record_tabular("sil_max_neg_logp", np.max(sil_nlogp))
            logger.dump_tabular()
            runner.show_heatmap()
    env.close()
    return model
