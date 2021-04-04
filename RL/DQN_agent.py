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

class agent:
    def __init__(self, input_dim, lam, gamma, lr, eps_start=0.7, eps_end=0.1, eps_decay=1000):
        self.lam = lam
        self.gamma = gamma
        self.policy_model = Model(input_dim)
        self.target_model = Model(input_dim)
        self.optimizer = optim.Adam(params=self.policy_model.parameters(), lr=lr)

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.steps = 0
        
        self.loss_1_list = []

    def optimize_model(self, replay_memory):
        sample_size = min(len(replay_memory), 10)
        transactions = replay_memory.sample(sample_size)
        loss_1 = optimize_model(self.policy_model, self.target_model, transactions,
         self.optimizer, self.lam,self.gamma)
        self.loss_1_list.append(loss_1)
        self.steps+=1


    def take_random_actions(self, out):
        actions = list(np.where(out>-1e7)[0])
        if len(actions)==0:
            return 0
        action = random.choice(actions)
        return action
    
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) *  np.exp(-1. * self.steps / self.eps_decay)
        
        state = getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        state = torch.FloatTensor([state])
        out = self.policy_model(state)['policy'].detach().numpy()[0]
        if random.random()<eps_threshold:
            action = self.take_random_actions(out)
        else:
            action = out.argmax()
        row = action//8
        col = action%8
        return row, col