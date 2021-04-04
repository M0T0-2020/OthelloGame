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
from Othello import Othello

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

def getState(board, changeable_Pos, Position_Row, Position_Col, Change_Position):
    state = np.zeros((3,8,8))
    for i in range(1,3):
        state[i-1,board==i]=1
    for i, (r, c) in enumerate(zip(Position_Row,Position_Col)):
        n = len(Change_Position[i][0])
        state[2,r,c]=n
    return state.tolist()

def getReward(board):
    reward_1 = len(board[board==1])
    reward_2 = len(board[board==2])
    return reward_1/100,  reward_2/100

def get_play_data(agent_1, agent_2):
    othello = Othello()

    first_states = deque()
    first_rewards = deque()
    first_actions = deque()

    second_states = deque()
    second_rewards = deque()
    second_actions = deque()

    board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.make()

    while not done:
        
        if othello.color==1:
            state = getState( board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            first_states.appendleft(state)
            reward_1, reward_2 = getReward(board)
            first_rewards.appendleft(reward_1)
            
            setrow, setcol = agent_1.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.step(setrow, setcol)
            
            first_actions.appendleft(8*setrow+setcol)
            
        else:
            state = getState( board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            second_states.appendleft(state)
            reward_1, reward_2 = getReward(board)
            second_rewards.appendleft(reward_2)
            
            setrow, setcol = agent_2.take_action(board, changeable_Pos, Position_Row, Position_Col, Change_Position)
            board, changeable_Pos, Position_Row, Position_Col, Change_Position, done = othello.step(setrow, setcol)
            
            second_actions.appendleft(8*setrow+setcol)

    state = getState( board, changeable_Pos, Position_Row, Position_Col, Change_Position)  
    reward_1, reward_2 = getReward(board)      
    first_states.appendleft(state)
    second_states.appendleft(state)
    if reward_1>reward_2:
        first_rewards.appendleft(reward_1*(100/32))
        second_rewards.appendleft(-reward_1*(100/32))
    elif reward_1<reward_2:
        first_rewards.appendleft(-reward_2*(100/32))
        second_rewards.appendleft(reward_2*(100/32))
    else:
        first_rewards.appendleft(0)
        second_rewards.appendleft(0)

    first_states = torch.FloatTensor(first_states)
    second_states = torch.FloatTensor(second_states)

    first_actions = torch.FloatTensor(first_actions)
    second_actions = torch.FloatTensor(second_actions)

    first_rewards = torch.FloatTensor(list(first_rewards)[:-1])
    second_rewards = torch.FloatTensor(list(second_rewards)[:-1])

    v = (len(board[board==1])+1)/(len(board[board==2])+1)
    first_v = 1 if v>1 else 0 if v==1 else -1
    second_v = -first_v

    first_values = torch.arange(0,len(first_rewards))
    first_values = first_values
    first_values = first_v*(first_values/first_values.max()).float()
    second_values = torch.arange(0,len(second_rewards))
    second_values = second_values
    second_values = second_v*(second_values/second_values.max()).float()

    data_first = {
        'states':first_states,
        'rewards':first_rewards,
        'actions':first_actions,
        'values':first_values,
    }

    data_secound = {
        'states':second_states,
        'rewards':second_rewards,
        'actions':second_actions,
        'values':second_values,
    }

    return data_first, data_secound


def test():
    agent_1 = randomAgent()
    agent_2 = greedyAgent()
    data_first, data_secound = get_play_data(agent_1, agent_2)

    for k in data_first.keys():
        print(f"{k} size {data_first[k].size()}")
        print(f"{k} size {data_secound[k].size()}")
        print(f"{data_first[k]}")
        print(f"{data_secound[k]}")