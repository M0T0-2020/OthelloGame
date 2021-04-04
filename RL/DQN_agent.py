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
from RL.utils import to_gpu_if_available
from RL.loss import *
from RL.optimize_model import optimize_dqncmodel as optimize_model
from RL.model import DQN_Model as Model
from RL.train_model import getState
from RL.noise import AdaptiveParamNoiseSpec, dqn_distance_metric

class agent:
    def __init__(self, input_dim, lam, gamma, lr, eps_start=0.7, eps_end=0.1, eps_decay=1000, noise=None):
        self.lam = lam
        self.gamma = gamma
        self.policy_model = Model(input_dim)
        self.target_model = Model(input_dim)
        self.optimizer = optim.Adam(params=self.policy_model.parameters(), lr=lr)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.noise = noise
        self.steps = 0
        
        self.loss_1_list = []

    def optimize_model_withNoise(self,replay_memory):
        sample_size = min(len(replay_memory), 5)
        transactions = replay_memory.sample(sample_size)
        loss_1 = optimize_model(self.policy_model, self.target_model, transactions,
        self.optimizer, self.lam,self.gamma)
        
        distance = dqn_distance_metric(transactions, self.policy_model, self.target_model)
        self.noise.adapt(distance)

        self.loss_1_list.append(loss_1)
        self.steps+=1

    def optimize_model_withoutNoise(self, replay_memory):
        sample_size = min(len(replay_memory), 10)
        transactions = replay_memory.sample(sample_size)
        loss_1 = optimize_model(self.policy_model, self.target_model, transactions,
         self.optimizer, self.lam,self.gamma)
        self.loss_1_list.append(loss_1)
        self.steps+=1

    def take_actions_withNoise(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *  np.exp(-1. * self.steps / self.eps_decay)
        perturbed_model = copy.deepcopy(self.policy_model)
        params = perturbed_model.state_dict()
        for name in params:
            if 'ln' in name: 
                pass 
            param = params[name]
            noise = torch.normal(mean=0, std=self.noise.current_stddev, size=param.shape)
            param += noise
            
        out = perturbed_model(state)['policy'].detach().numpy()[0]
        if random.random()<eps_threshold:
            action = self.take_random_actions(out)
        else:
            action = out.argmax()
        return action
    
    def take_actions_withoutNoise(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *  np.exp(-1. * self.steps / self.eps_decay)
        out = self.policy_model(state)['policy'].detach().numpy()[0]
        if random.random()<eps_threshold:
            action = self.take_random_actions(out)
        else:
            action = out.argmax()
        return action

    def take_random_actions(self, out):
        actions = list(np.where(out>-1e7)[0])
        if len(actions)==0:
            return 0
        action = random.choice(actions)
        return action
    
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        state = getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        state = torch.FloatTensor([state])
        if self.noise is not None:
            action = self.take_actions_withNoise(state)
        else:
            action = self.take_actions_withoutNoise(state)
        row = action//8
        col = action%8
        return row, col

    def take_determ_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        state = getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        state = torch.FloatTensor([state])
        out = self.policy_model(state)['policy'].detach().numpy()[0]
        action = out.argmax()
        row = action//8
        col = action%8
        return row, col