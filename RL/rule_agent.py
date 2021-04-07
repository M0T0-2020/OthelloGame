import os, sys
import time
import copy
import random
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class randomAgent:
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        if len(Change_Position)==0:
            setrow, setcol = 0,0
        else:
            idx = np.random.randint(len(Position_Row))
            setrow, setcol = Position_Row[idx], Position_Col[idx]
        return setrow, setcol
    
class greedyAgent:
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        if len(Change_Position)==0:
            setrow, setcol = 0,0
        else:
            Change_Position_len = [len(l[0]) for l in Change_Position]
            idx_l = [i for i, l in enumerate(Change_Position_len) if l==max(Change_Position_len)]
            idx = np.random.choice(idx_l)
            setrow, setcol = Position_Row[idx], Position_Col[idx]
        return setrow, setcol

class RollOutAgent:
    def __init__(self, agent_1, agent_2, n=16):
        self.rollout_num = n
        self.agent_1 = agent_1        
        self.agent_2 = agent_2
        self.steps = 0
        
    def take_action(self, board, changeable_Pos, Position_Row, Position_Col, Change_Position):
        if self.steps<self.rollout_num:
            action = self.agent_1.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        else:
            action = self.agent_2.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
        self.steps+=1
        return action