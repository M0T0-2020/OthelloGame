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
from torch.distributions import Categorical

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL.loss import *
from RL.optimize_model import optimize_dqncmodel as optimize_model
from RL.model import DQN_Model as Model
from RL.train_model3 import randomAgent, greedyAgent, get_play_data, getState
from RL.noise import AdaptiveParamNoiseSpec, dqn_distance_metric
from RL.sam import SAM
from RL.utils import get_device

class agent:
    def __init__(self, input_dim, lam, gamma, lr, target_update=7, eps_start=0.7, eps_end=0.1, eps_decay=1000, noise=None):
        self.lam = lam
        self.gamma = gamma
        self.policy_model = Model(input_dim)
        self.target_model = Model(input_dim)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.perturbed_model = copy.deepcopy(self.policy_model)
        #self.optimizer = optim.Adam(params=self.policy_model.parameters(), lr=lr)
        
        base_optimizer = torch.optim.Adam  # define an optimizer for the "sharpness-aware" update
        self.optimizer = SAM(self.policy_model.parameters(), base_optimizer, lr=lr)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.noise = noise
        self.steps = 0
        self.target_update = target_update
        
        self.loss_1_list = []

    def init_param_optim(self, n=100):
        device = get_device()
        self.policy_model = self.policy_model.to(device)
        agent_1 = randomAgent()
        agent_2 = greedyAgent()
        tmp_optimizer = optim.Adam(params=self.policy_model.parameters(), lr=1e-3)
        tmp_loss_list = []
        for _ in range(n):
            states = []
            for _ in range(4):
                data_1, data_2 = get_play_data(agent_1, agent_2)
                states.append(data_1['states'])
                states.append(data_2['states'])
            states = torch.cat(states, dim=0).to(device)
            policy = self.policy_model(states)['policy']
            y = states[:,2,:,:].flatten(1)
            y[y>0]=1
            y *= 30
            loss = ((policy-y)**2).mean()
            tmp_optimizer.zero_grad()
            loss.backward()
            tmp_optimizer.step()
            tmp_loss_list.append(loss.item())
        self.policy_model = self.policy_model.to('cpu')
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.perturbed_model = copy.deepcopy(self.policy_model)
        return tmp_loss_list

    def optimize_model(self,replay_memory):
        sample_size = min(len(replay_memory), 2**3)
        transactions = replay_memory.sample(sample_size)
        loss_1 = optimize_model(self.policy_model, self.target_model, transactions,
        self.optimizer, self.lam,self.gamma)
        if self.noise is not None:
            distance = dqn_distance_metric(transactions, self.policy_model, self.perturbed_model)
            self.noise.adapt(distance)
        self.loss_1_list.append(loss_1)
        self.steps+=1
        if self.steps%self.target_update==0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

    def take_actions_withNoise(self, state, changeable_p):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *  np.exp(-1. * self.steps / self.eps_decay)
        self.perturbed_model = copy.deepcopy(self.policy_model)
        params = self.perturbed_model.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            noise = torch.normal(mean=0, std=self.noise.current_stddev, size=param.shape)
            param += noise
            
        out = self.perturbed_model(state)['policy'].detach().numpy()[0]
        if random.random()<eps_threshold:
            action = self.take_random_actions(changeable_p)
            return action
        else:
            actions = out.argsort()[::-1]
            for action in actions:
                if action in changeable_p:
                    return action
            return 0
    
    def take_actions_withoutNoise(self, state, changeable_p):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *  np.exp(-1. * self.steps / self.eps_decay)
        out = self.policy_model(state)['policy'].detach().numpy()[0]
        if random.random()<eps_threshold:
            action = self.take_random_actions(changeable_p)
            return action
        else:
            actions = out.argsort()[::-1]
            for action in actions:
                if action in changeable_p:
                    return action
            return 0

    def take_random_actions(self, changeable_p):
        if len(changeable_p)==0:
            return 0
        action = random.choice(changeable_p)
        return action
    
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        state = getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        state = torch.FloatTensor([state])
        changeable_p = list(np.where(changeable_Pos.flatten()>0)[0])
        if self.noise is not None:
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
        out = self.policy_model(state)['policy'].detach().numpy()[0]
        actions = out.argsort()[::-1]
        for action in actions:
            if action in changeable_p:
                row = action//8
                col = action%8
                return row, col
        row = action//8
        col = action%8
        return row, col