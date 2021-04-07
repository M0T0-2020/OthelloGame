import os, sys
import time
import copy
import random
import pickle
from collections import deque
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL.loss import *
from RL.optimize_model import optimize_a2cmodel as optimize_model
from RL.model import A2C_Model as Model
from RL.train_model import randomAgent, greedyAgent, get_play_data, getState
from RL.sam import SAM
from RL.utils import get_device

class agent:
    def __init__(self, input_dim=3, lam=0.6, gamma=0.9, lr=1e-4, weight_decay=1e-5, noise=None):
        self.lam = lam
        self.gamma = gamma
        self.model = Model(input_dim)
        self.perturbed_model = copy.deepcopy(self.model)
        base_optimizer = optim.Adam
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, weight_decay=weight_decay)
        self.noise = noise

        self.steps = 0
        self.loss_1_list = []
        self.loss_2_list = []
        self.loss_3_list = []

    def init_param_optim(self, n=100):
        device = get_device()
        self.model = self.model.to(device)
        agent_1 = randomAgent()
        agent_2 = greedyAgent()
        tmp_optimizer = optim.Adam(params=self.model.parameters(), lr=1e-3)
        tmp_loss_list = []
        for _ in range(n):
            states = []
            for _ in range(4):
                data_1, data_2 = get_play_data(agent_1, agent_2)
                states.append(data_1['states'])
                states.append(data_2['states'])
            states = torch.cat(states, dim=0).to(device)
            policy = self.model(states)['policy']
            y = states[:,2,:,:].flatten(1)
            y[y>0]=1
            y *= 30
            loss = ((policy-y)**2).mean()
            tmp_optimizer.zero_grad()
            loss.backward()
            tmp_optimizer.step()
            tmp_loss_list.append(loss.item())
        self.model = self.model.to('cpu')
        self.perturbed_model.load_state_dict(self.model.state_dict())
        return tmp_loss_list

    def a2c_distance_metric(self, transactions, model, perturbed_model):
        device = get_device()
        model = model.to(device)
        perturbed_model = perturbed_model.to(device)
        states = []
        for transaction in transactions:
            states.append(copy.deepcopy(transaction['states']))
        states = torch.cat(states).to(device)
        out = model(states)
        perturbed_out = perturbed_model(states)
        distance=0
        for key in out.keys():
            distance += nn.MSELoss()(out[key], perturbed_out[key]).detach()
        model = model.to('cpu')
        perturbed_model = perturbed_model.to('cpu')
        return distance

    def optimize_model(self, replay_memory):
        sample_size = min(len(replay_memory), 10)
        transactions = replay_memory.sample(sample_size)
        loss_1, loss_2, loss_3 = optimize_model(self.model, transactions, self.optimizer, step=self.steps)
        self.loss_1_list.append(loss_1)
        self.loss_2_list.append(loss_2)
        self.loss_3_list.append(loss_3)
        self.steps+=1

        if self.noise is not None:
            distance = self.a2c_distance_metric(transactions, self.model, self.perturbed_model)
            self.noise.adapt(distance)

    def take_actions_withNoise(self, state, changeable_p):
        self.perturbed_model = copy.deepcopy(self.model)
        params = self.perturbed_model.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            noise = torch.normal(mean=0, std=self.noise.current_stddev, size=param.shape)
            param += noise
            
        out = self.perturbed_model(state)['policy'].detach()[0]
        out = out[changeable_p]
        sample = Categorical(logits=out).sample()
        action = changeable_p[sample]
        return action

    def take_actions_withoutNoise(self, state, changeable_p):
        out = self.model(state)['policy'].detach()[0]
        out = out[changeable_p]
        sample = Categorical(logits=out).sample()
        action = changeable_p[sample]
        return action

    def take_random_actions(self, changeable_p):
        action = random.choice(changeable_p)
        return action
    
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        state = getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        state = torch.FloatTensor([state])
        changeable_p = list(np.where(changeable_Pos.flatten()>0)[0])
        if len(changeable_p)==0:
            action = 0
        elif self.noise is not None:
            action = self.take_actions_withNoise(state, changeable_p)
        else:
            action = self.take_actions_withoutNoise(state, changeable_p)
        row = action//8
        col = action%8
        return row, col

    def take_determ_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        state = getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        changeable_p = list(np.where(changeable_Pos.flatten()>0)[0])
        state = torch.FloatTensor([state])
        out = self.model(state)['policy'].detach().numpy()[0]
        actions = out.argsort()[::-1]
        for action in actions:
            if action in changeable_p:
                row = action//8
                col = action%8
                return row, col
        row = action//8
        col = action%8
        return row, col