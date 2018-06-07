import numpy as np
import tensorflow as tf
import random
from gym import spaces

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
from baselines.a2c.utils import discount_with_dones


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, obs_t, action, R):
        data = (obs_t, action, R)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, returns= [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, R = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            returns.append(R)
        return np.array(obses_t), np.array(actions), np.array(returns)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        R_batch: np.array
            returns received as results of executing act_batch
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """

        idxes = self._sample_proportional(batch_size)
        
        if beta > 0:
            weights = []
            p_min = self._it_min.min() / self._it_sum.sum()
            max_weight = (p_min * len(self._storage)) ** (-beta)

            for idx in idxes:
                p_sample = self._it_sum[idx] / self._it_sum.sum()
                weight = (p_sample * len(self._storage)) ** (-beta)
                weights.append(weight / max_weight)
            weights = np.array(weights)
        else:
            weights = np.ones_like(idxes, dtype=np.float32)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class SelfImitation(object):
    
    def __init__(self, model_ob, model_vf, model_entropy, 
            fn_value, fn_neg_log_prob, ac_space, fn_reward, fn_obs=None,
            n_env=16, batch_size=512, n_update=4, 
            clip=1, w_value=0.01, w_entropy=0.01, 
            max_steps=int(1e5), gamma=0.99,
            max_nlogp=5, min_batch_size=64, stack=1,
            alpha=0.6, beta=1.0):

        self.model_ob = model_ob
        self.model_vf = model_vf
        self.model_entropy = model_entropy
        self.fn_value = fn_value
        self.fn_neg_log_prob = fn_neg_log_prob
        self.fn_reward = fn_reward
        self.fn_obs = fn_obs

        self.beta = beta
        self.buffer = PrioritizedReplayBuffer(max_steps, alpha)
        self.n_env = n_env
        self.batch_size = batch_size
        self.n_update = n_update
        self.clip = clip
        self.w_loss = 1.0
        self.w_value = w_value
        self.w_entropy = w_entropy
        self.max_steps = max_steps
        self.gamma = gamma
        self.max_nlogp = max_nlogp
        self.min_batch_size = min_batch_size

        self.stack = stack
        self.train_count = 0
        self.update_count = 0
        self.total_steps = []
        self.total_rewards = []
        self.running_episodes = [[] for _ in range(n_env)]

        if isinstance(ac_space, spaces.Box):
            # Continuous control
            self.A = tf.placeholder(tf.float32, [None, ac_space.shape[0]])
        elif isinstance(ac_space, spaces.Discrete):
            # Discrete control
            self.A = tf.placeholder(tf.int32, [None])
        else:
            raise NotImplementedError

        self.R = tf.placeholder(tf.float32, [None])
        self.W = tf.placeholder(tf.float32, [None])
        self.build_loss_op()

    def set_loss_weight(self, w):
        self.w_loss = w

    def add_episode(self, trajectory):
        obs = []
        actions = []
        rewards = []
        dones = []

        if self.stack > 1:
            ob_shape = list(trajectory[0][0].shape)
            nc = ob_shape[-1]
            ob_shape[-1] = nc*self.stack
            stacked_ob = np.zeros(ob_shape, dtype=trajectory[0][0].dtype)
        for (ob, action, reward) in trajectory:
            if ob is not None:
                x = self.fn_obs(ob) if self.fn_obs is not None else ob
                if self.stack > 1:
                    stacked_ob = np.roll(stacked_ob, shift=-nc, axis=2)
                    stacked_ob[:, :, -nc:] = x
                    obs.append(stacked_ob)
                else:
                    obs.append(x)
            else:
                obs.append(None)
            actions.append(action)
            rewards.append(self.fn_reward(reward))
            dones.append(False)
        dones[len(dones)-1]=True
        returns = discount_with_dones(rewards, dones, self.gamma)
        for (ob, action, R) in list(zip(obs, actions, returns)):
            self.buffer.add(ob, action, R)

    def update_buffer(self, trajectory):
        positive_reward = False
        for (ob, a, r) in trajectory:
            if r > 0:
                positive_reward = True
                break
        if positive_reward:
            self.add_episode(trajectory)
            self.total_steps.append(len(trajectory))
            self.total_rewards.append(np.sum([x[2] for x in trajectory]))
            while np.sum(self.total_steps) > self.max_steps and len(self.total_steps) > 1:
                self.total_steps.pop(0)
                self.total_rewards.pop(0)

    def num_steps(self):
        return len(self.buffer)

    def num_episodes(self):
        return len(self.total_rewards)

    def get_best_reward(self):
        if len(self.total_rewards) > 0:
            return np.max(self.total_rewards)
        return 0

    def step(self, obs, actions, rewards, dones):
        for n in range(self.n_env):
            if self.n_update > 0:
                self.running_episodes[n].append([obs[n], actions[n], rewards[n]])
            else:
                self.running_episodes[n].append([None, actions[n], rewards[n]])

        for n, done in enumerate(dones):
            if done:
                self.update_buffer(self.running_episodes[n])
                self.running_episodes[n] = []

    def sample_batch(self, batch_size):
        if len(self.buffer) > 100:
            batch_size = min(batch_size, len(self.buffer))
            return self.buffer.sample(batch_size, beta=self.beta)
        else:
            return None, None, None, None, None

    def build_loss_op(self):
        mask = tf.where(self.R - tf.squeeze(self.model_vf) > 0.0, 
                tf.ones_like(self.R), tf.zeros_like(self.R))
        self.num_valid_samples = tf.reduce_sum(mask)
        self.num_samples = tf.maximum(self.num_valid_samples, self.min_batch_size)
       
        # Policy update
        nlogp = self.fn_neg_log_prob(self.A)
        clipped_nlogp = tf.stop_gradient(
                tf.minimum(nlogp, self.max_nlogp) - nlogp) + nlogp
        
        # clipped_nlogp = nlogp
        self.neg_log_p = clipped_nlogp
        self.adv = tf.stop_gradient(
                tf.clip_by_value(self.R - tf.squeeze(self.model_vf), 0.0, self.clip))
        self.mean_adv = tf.reduce_sum(self.adv) / self.num_samples
        self.pg_loss = tf.reduce_sum(self.W * self.adv * clipped_nlogp) / self.num_samples

        # Entropy regularization
        entropy = tf.reduce_sum(self.W * self.model_entropy * mask) / self.num_samples
        self.loss = self.pg_loss - entropy * self.w_entropy
            
        # Value update
        v_target = self.R
        v_estimate = tf.squeeze(self.model_vf)
        delta = tf.clip_by_value(v_estimate - v_target, -self.clip, 0) * mask
        self.vf_loss = tf.reduce_sum(self.W * v_estimate * tf.stop_gradient(delta)) / self.num_samples
        self.loss += 0.5 * self.w_value * self.vf_loss
        return self.loss

    def build_train_op(self, params, optim, lr, max_grad_norm=0.5):
        self.LR = lr
        grads = tf.gradients(self.loss * self.w_loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        self.train_op = optim.apply_gradients(grads)

    def _train(self, sess, lr):
        obs, actions, returns, weights, idxes = self.sample_batch(self.batch_size)
        if obs is None:
            return 0, 0, 0, 0

        loss, adv, mean_adv, samples, nlogp, _ = sess.run(
                [self.loss, self.adv, self.mean_adv, self.num_valid_samples,
                    self.neg_log_p, self.train_op], 
                {self.model_ob: obs, 
                 self.A: actions, 
                 self.R: returns,
                 self.LR: lr,
                 self.W: weights})       

        self.buffer.update_priorities(idxes, adv)
        return loss, mean_adv, samples, nlogp

    def train(self, sess, lr):
        if self.n_update == 0:
            return 0, 0, 0, 0

        self.train_count += 1
        loss, adv, samples, nlogp = 0, 0, 0, 0
        if self.n_update < 1:
            update_ratio = int(1/self.n_update + 1e-8)
            if self.train_count % update_ratio == 0:
                loss, adv, samples, nlogp = self._train(sess, lr)
        else: # n_update > 1 
            for n in range(int(self.n_update)):
                loss, adv, samples, nlogp = self._train(sess, lr)

        return loss, adv, samples, nlogp
