"""This module is main module for contestant's solution."""

from hackathon.utils.control import Control
from hackathon.utils.utils import ResultsMessage, DataMessage, PVMode, \
    TYPHOON_DIR, config_outs
from hackathon.framework.http_server import prepare_dot_dir
import gym
import hackathon.solution.smarthomeenv.smarthomeenv


from hackathon.solution.learning.dl.dqn import DQN
from hackathon.solution.learning.dl.dqn.dqn import to_tensor
from hackathon.solution.smarthomeenv.smarthomeenv.envs import SmartHomeEnv
import numpy as np
import torch

env = gym.make('smarthomeenv-v0')
actions = SmartHomeEnv().actions
env.reset()
net = DQN(40)

if torch.cuda.is_available():
    net = net.cuda()
    net.load_state_dict(torch.load('hackathon/solution/model.pytorch'))
else:
    net.load_state_dict(torch.load('hackathon/solution/model.pytorch', map_location='cpu'))


def worker(msg: DataMessage) -> ResultsMessage:
    action = np.argmax(net.forward(to_tensor(msg)).detach().numpy())
    real_action = actions[action]
    real_action.data_msg = msg

    if msg.solar_production > 0:
        real_action.pv_mode = PVMode.ON

    return real_action


def run(args) -> None:
    prepare_dot_dir()
    config_outs(args, 'solution')

    cntrl = Control()

    for data in cntrl.get_data():
        cntrl.push_results(worker(data))
