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
    return reward_1/40,  reward_2/40

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
    first_states = torch.FloatTensor(first_states)
    second_states = torch.FloatTensor(second_states)

    first_actions = torch.FloatTensor(first_actions)
    second_actions = torch.FloatTensor(second_actions)

    first_rewards.appendleft(reward_1)
    second_rewards.appendleft(reward_2)
    first_rewards = torch.FloatTensor(list(first_rewards)[:-1])
    second_rewards = torch.FloatTensor(list(second_rewards)[:-1])

    discount_rate=1
    if reward_1>reward_2:
        first_values = torch.FloatTensor([discount_rate**i for i in range(len(first_rewards))])
        second_values = torch.FloatTensor([-discount_rate**i for i in range(len(second_rewards))])
    elif reward_1<reward_2:
        first_values = torch.FloatTensor([-discount_rate**i for i in range(len(first_rewards))])
        second_values = torch.FloatTensor([discount_rate**i for i in range(len(second_rewards))])
    else:
        first_values = torch.FloatTensor([0 for i in range(len(first_rewards))])
        second_values = torch.FloatTensor([0 for i in range(len(second_rewards))])

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