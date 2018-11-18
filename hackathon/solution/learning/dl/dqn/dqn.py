import os
from datetime import datetime
from random import random, randrange

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

import gym

from tqdm import tqdm

from hackathon.utils.utils import DataMessage
from hackathon.solution.learning.dl import Memory, Transition

cuda_available = torch.cuda.is_available()


def to_tensor(lazy_frame):
    if isinstance(lazy_frame, DataMessage):
        lazy_frame = list(lazy_frame.__dict__.values())
        lazy_frame = lazy_frame[1:] # sklonio id

    return torch.from_numpy(np.array(lazy_frame))


def variable(t: torch.Tensor, **kwargs):
    if cuda_available:
        t = t.cuda()
    return Variable(t, **kwargs)


class DQN(nn.Module):
    def __init__(self, output_size: int):
        """
        :param output_size: actions
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 128)
        self.output = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

    @staticmethod
    def loss(output, target, *args):
        assert isinstance(output, Variable) and isinstance(target, Variable)
        return F.smooth_l1_loss(output, target.reshape(32, 1), size_average=False)


class Agent(object):
    def __init__(self, env: gym.Env, gamma, epsilon, final_epsilon, final_exp_step):
        """
        :param env: environment
        :param gamma: discount rate
        :param epsilon: initial exploration rate
        :param final_epsilon: final exploration rate
        :param final_exp_step: the step terminating exploration
        """
        self.env = env
        self.action_size = self.env.action_space.n
        self.net = DQN(self.action_size)
        self.target_net = DQN(self.action_size)
        self._gamma = gamma
        self._initial_epsilon = epsilon
        self.epsilon = epsilon
        self._final_epsilon = final_epsilon
        self._final_exp_step = final_exp_step
        if cuda_available:
            self.net.cuda()
            self.target_net.cuda()
        self.update_target_net()

    def policy(self, state, epsilon):
        """
        epsilon greedy ploicy
        :param state: np.ndarray
        :param epsilon: exploration rate
        :return: action for given state
        """
        if random() <= epsilon:
            action = randrange(0, self.action_size)
        else:
            state = variable(to_tensor(state).unsqueeze(0))
            action = self.net(state).data.view(-1).max(dim=0)[1].sum()
        return action

    def parameter_scheduler(self, step):
        """
        \epsilon_t =epsilon_0 + t * \frac{\epsilon_T-\epsilon_0}{\epsilon_0}
        """
        if step < self._final_exp_step:
            self.epsilon = step * (
                    self._final_epsilon - self._initial_epsilon) / self._final_exp_step + self._initial_epsilon

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

    def estimate_value(self, reward, state, done):
        q_hat = self.target_net(variable(state)).max(dim=1)[0]
        estimated = variable(reward) + self._gamma * variable(done) * q_hat
        return estimated

    def q_value(self, state, action):
        return self.net(variable(state)).gather(1, variable(action.unsqueeze(dim=1)))

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)


class Trainer(object):
    def __init__(self, agent: Agent, val_env: gym.Env, lr, memory_size, target_update_freq, gradient_update_freq,
                 batch_size, replay_start, val_freq, log_freq_by_step, log_freq_by_ep, val_epsilon,
                 log_dir, weight_dir):
        """
        :param agent: agent object
        :param val_env: environment for validation
        :param lr: learning rate of optimizer
        :param memory_size: size of replay memory
        :param target_update_freq: frequency of update target network in steps
        :param gradient_update_freq: frequency of q-network update in steps
        :param batch_size: batch size for q-net
        :param replay_start: number of random exploration before starting
        :param val_freq: frequency of validation in steps
        :param log_freq_by_step: frequency of logging in steps
        :param log_freq_by_ep: frequency of logging in episodes
        :param val_epsilon: exploration rate for validation
        :param log_dir: directory for saving tensorboard things
        :param weight_dir: directory for saving weights when validated
        """
        self.agent = agent
        self.env = self.agent.env
        self.val_env = val_env
        self.optimizer = optim.RMSprop(params=self.agent.net.parameters(), lr=lr)
        self.memory = Memory(memory_size)
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.replay_start = replay_start
        self.gradient_update_freq = gradient_update_freq
        self._step = 0
        self._episode = 0
        self._warmed = False
        self._val_freq = val_freq
        self.log_freq_by_step = log_freq_by_step
        self.log_freq_by_ep = log_freq_by_ep
        self._val_epsilon = val_epsilon

        if weight_dir is not None and not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        self.weight_dir = weight_dir

    def warm_up(self):
        # to populate replay memory
        state_before = self.env.reset()
        self._warmed = True
        for _ in tqdm(range(self.replay_start)):
            action = self.env.action_space.sample()
            state_after, reward, done, _ = self.env.step(action)
            self.memory(Transition(state_before, action, reward, state_after, done))
            state_before = self.env.reset() if done else state_after

    def _train_nn(self):
        # neural network part
        self.optimizer.zero_grad()
        batch_state_before, batch_action, batch_reward, batch_state_after, batch_done = self.get_batch()
        target = self.agent.estimate_value(batch_reward, batch_state_after, batch_done)
        q_value = self.agent.q_value(batch_state_before, batch_action)
        loss = self.agent.net.loss(q_value, target)
        if self._step % self.gradient_update_freq == 0:
            loss.backward()
            self.optimizer.step()

        return loss.data.item()

    def _loop(self, is_train):
        # internal loop for both training and validation
        done = False
        state_before = self.env.reset() if is_train else self.val_env.reset()
        loss_list = []
        reward_list = []
        while not done:
            epsilon = self.agent.epsilon if is_train else self._val_epsilon
            action = self.agent.policy(state_before, epsilon)
            state_after, reward, done, _ = self.env.step(action) if is_train else self.val_env.step(action)

            if is_train:
                self._step += 1
                self.memory(Transition(state_before, action, reward, state_after, done))
                self.agent.parameter_scheduler(self._step)
                loss_list.append(self._train_nn())

            state_before = state_after
            reward_list.append(reward)
            if self._step % self.target_update_freq == 0 and is_train:
                self.agent.update_target_net()

            if self._step % self._val_freq == 0 and is_train:
                self.val()

        return loss_list, reward_list, state_after

    def train(self, max_step):
        if not self._warmed:
            self.warm_up()

        try:
            while self._step < max_step:
                self._episode += 1
                train_loss, train_reward, _state = self._loop(is_train=True)

                if self._episode % self.log_freq_by_ep == 0:
                    print(f"episode: {self._episode:>5}/step: {self._step:>6}/"
                          f"loss: {np.mean(train_loss):>7.2f}/reward: {sum(train_reward):.2f}/size: {len(train_loss)}")

        except KeyboardInterrupt as ke:
            print(ke)
        finally:
            self.agent.save('model.pytorch')
            print('pytorch model saved')

    def val(self):
        # validation
        _, val_reward, _ = self._loop(is_train=False)

        if self.weight_dir is not None:
            self.agent.save(os.path.join(self.weight_dir, f"{self._step}.pkl"))

    def get_batch(self):
        # return batch
        batch = self.memory.sample(self.batch_size)
        batch_state_before = torch.cat([to_tensor(m.state_before).unsqueeze(0) for m in batch], dim=0)
        batch_action = torch.LongTensor([m.action for m in batch])
        batch_reward = torch.Tensor([m.reward for m in batch])
        batch_state_after = torch.cat([to_tensor(m.state_after).unsqueeze(0) for m in batch], dim=0)
        # tensor: 0 if done else 1
        batch_done = 1 - torch.Tensor([m.done for m in batch])
        return batch_state_before, batch_action, batch_reward, batch_state_after, batch_done
