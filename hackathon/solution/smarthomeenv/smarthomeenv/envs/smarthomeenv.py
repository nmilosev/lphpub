import gym
from gym import spaces
import json

from hackathon.solution.smarthomeenv.smarthomeenv.envs.rating import get_physics_metrics, reset_count
from hackathon.utils.utils import DataMessage, ResultsMessage, PVMode


class SmartHomeEnv(gym.Env):
    def reset(self):
        self.__init__()
        reset_count()
        return self.states[0]

    def render(self, mode='human'):
        print('ispis na konzolu')

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.states = []

        import itertools as it

        my_dict = {'1_loadOne': [True], '2_loadTwo': [True, False], '3_loadThree': [True, False],
                   '4_powerRef': [-5.0, -2.5, 0.0, 2.5, 5.0], '5_pvmode': [PVMode.OFF, PVMode.ON]}
        combinations = it.product(*(my_dict[name] for name in my_dict.keys()))
        combs = list(combinations)

        self.actions = []
        self.states = []

        for comb in combs:
            self.actions.append(ResultsMessage(None, *comb))

        with open('data/profiles.json', 'r') as f, open('data/physics_init.json') as f2:
            physics = json.loads(f2.read())
            state_id = 0
            for d in json.loads(f.read()):
                self.states.append(DataMessage(id=state_id,
                                               bessOverload=physics['bessOverload'],
                                               bessPower=physics['bessPower'],
                                               bessSOC=physics['bessSOC'],
                                               mainGridPower=physics['mainGridPower'],
                                               grid_status=d['gridStatus'],
                                               buying_price=d['buyingPrice'],
                                               selling_price=d['sellingPrice'],
                                               solar_production=d['solarProduction'],
                                               current_max_load=d['currentLoad']))
                state_id += 1

        self.current_id = 0
        self.current = []
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(7200 * len(self.actions))

    def step(self, action):

        new_state, energy_use, penal = \
            self.get_a_result(*get_physics_metrics(self.states[self.current_id],
                                                   self.actions[action]),
                              self.states[self.current_id])

        self.current_id += 1

        self.states[self.current_id].bessSOC = new_state['bessSOC']
        self.states[self.current_id].bessPower = new_state['bessPower']
        self.states[self.current_id].bessOverload = new_state['bessOverload']
        self.states[self.current_id].mainGridPower = new_state['mainGridPower']

        # Nemanjino:
        # reward = 343 + new_state['bessSOC'] / \
        #          (new_state['energyMark'] if new_state['energyMark'] > 0 else 1) \
        #          * (new_state['penal'] ** 4) * (1 + new_state['bessSOC'] < 0.25) \
        #          + new_state['bessOverload'] * 3

        # staro:
        # reward = 0
        # if energy_use == 0 or new_state['bessSOC'] < 0.33: # sve je ugasio idiot
        #     reward = 0
        # elif energy_use < 0:  # ovo znaci da prodaje
        #     reward = 1
        # elif penal == 0:
        #     reward = 10
        # elif penal <= 0.7:
        #     reward = 3
        # else:  # if energy_use >= 0: # kupuje struju i ima velik penal
        #     reward = 0.5
        # Dimitrije:

        if new_state['bessOverload'] or (self.states[self.current_id].grid_status == 0 and new_state['bessPower'] >= 0):
            reward = 0
        elif new_state['bessSOC'] < 0.33:
            reward = 5
        elif energy_use < 0:
            reward = 10
        elif penal == 0:
            reward = 100
        elif penal <= 0.7:
            reward = 20
        else:
            reward = 5

        # Nemanja 3:
        # if new_state['bessOverload'] or (self.states[self.current_id].grid_status == 0 and new_state['bessPower'] >= 0):
        #     reward = 0
        # else:
        #     reward = new_state['bessSOC'] * 100 / (1 + new_state['energyMark'] + new_state['penal']) ** 2

        #Jovic:

        # reward = 0

        # if new_state['bessOverload']:
        #     reward = 0
        # elif new_state['bessSOC'] < 0.25:
        #     reward = 5
        # elif energy_use < 0:
        #     reward = 10
        # elif penal == 0:
        #     reward = 100
        # elif 6.9 < new_state['mainGridPower'] < 9.7:
        #     if new_state['pv_power'] > 0:
        #         if new_state['bessSOC'] > 0.5:
        #             reward = 23
        #         else:
        #             reward = 17.1
        #     else:
        #         reward = 17
        # elif penal <= 0.7:
        #     reward = 20
        # elif penal < 1.7:
        #     reward = 30
        # else:
        #     reward = 5

        return self.states[self.current_id], reward, self.current_id == len(self.states) - 1, {}

    def get_a_result(self, energy_mark: float, performance_mark: float,
                     mg: float, penal: float, r_load: float, pv_power: float,
                     soc_bess: float, overload: bool, current_power: float,
                     data_msg: DataMessage):
        current_mark = energy_mark + performance_mark + penal
        last = self.current[-1]['overall'] if self.current else 0
        last_energy = self.current[-1]['overall_energy'] if self.current else 0
        last_penalty = self.current[-1]['overall_penalty'] if self.current else 0
        last_performance = self.current[-1]['overall_performance'] if self.current else 0
        new = {'overall': last + current_mark,
               'overall_energy': last_energy + energy_mark,
               'overall_penalty': last_penalty + penal,
               'overall_performance': last_performance + performance_mark,
               'energyMark': energy_mark,
               'performance': performance_mark,
               'real_load': r_load,
               'pv_power': pv_power,
               'bessSOC': soc_bess,
               'bessOverload': overload,
               'bessPower': current_power,
               'mainGridPower': mg,
               'penal': penal,
               'DataMessage': data_msg.__dict__}
        self.current.append(new)
        global LATEST_RESULT
        LATEST_RESULT = new

        return new, energy_mark, penal
