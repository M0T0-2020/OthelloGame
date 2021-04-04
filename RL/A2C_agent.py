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
from RL.optimize_model import optimize_a2cmodel as optimize_model
from RL.model import A2C_Model as Model
from RL.train_model import getState

class agent:
    def __init__(self, input_dim, lam, gamma, lr):
        self.lam = lam
        self.gamma = gamma
        self.model = Model(input_dim)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        self.steps = 0

        self.loss_1_list = []
        self.loss_2_list = []
        self.loss_3_list = []

    def optimize_model(self, replay_memory):
        sample_size = min(len(replay_memory), 10)
        transactions = replay_memory.sample(sample_size)
        loss_1, loss_2, loss_3, _ = optimize_model(self.model, transactions, self.optimizer)
        self.loss_1_list.append(loss_1)
        self.loss_2_list.append(loss_2)
        self.loss_3_list.append(loss_3)
        self.steps+=1

    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        state = getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        state = torch.FloatTensor([state])
        out = self.model(state)
        #x_2 = copy.deepcopy(state[:,2,:,:].flatten(1))
        softmax_func = nn.Softmax(1)
        softmax_policy = softmax_func(out['policy'].detach())
        #softmax_policy[x_2==0]=0
        softmax_policy = Categorical(probs=softmax_policy)
        action = softmax_policy.sample()[0]
        row = action//8
        col = action%8
        return row, col