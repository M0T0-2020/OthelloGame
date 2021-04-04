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

def optimize_a2cmodel(model, transactions, optimizer, gamma=.9, lam=.6):
    model = to_gpu_if_available(model)

    policy_loss = []
    value_loss = []
    entropy = []
    othello_loss = 0

    for transaction in transactions:
        states = copy.deepcopy(transaction['states'])
        rewards = copy.deepcopy(transaction["rewards"])
        actions = copy.deepcopy(transaction["actions"])
        
        out = model(states)
        policy = out['policy'][1:]
        policy = Categorical(logits=policy)
        value = out['value'][1:]
        target_value = out['value'][:-1].detach()
        
        td_target_value = calu_td_target_value_a2c(target_value, rewards, gamma, lam)
        ploss, vloss, ent, oth = calu_loss(policy, actions, value, td_target_value, states)

        policy_loss+=ploss
        value_loss+=vloss
        entropy+=ent
        othello_loss+=oth/len(transactions)
    
    policy_loss = torch.cat(policy_loss, dim=0).mean()
    value_loss = torch.cat(value_loss, dim=0).mean()
    entropy = torch.cat(entropy, dim=0).mean()

    total_loss = policy_loss+2*value_loss+0.2*entropy#+0.01*othello_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    model.eval()
    model = model.to('cpu')

    return policy_loss.item(), value_loss.item(), entropy.item(), othello_loss.item()

def optimize_dqncmodel(policy_model, target_model, transactions, optimizer, gamma=.9, lam=.6):
    policy_model = to_gpu_if_available(policy_model)
    target_model = to_gpu_if_available(target_model)

    total_loss = []

    for transaction in transactions:
        states = copy.deepcopy(transaction['states'])
        rewards = copy.deepcopy(transaction["rewards"])
        actions = copy.deepcopy(transaction["actions"])
        
        policy_q = policy_model(states[1:])['policy']
        policy_q = policy_q.gather(1, actions.to(torch.int64).unsqueeze(1))

        target_q = target_model(states[:-1])['policy'].detach()
        td_target_value = calu_td_target_value_dqn(target_q, rewards, gamma, lam)
        loss = ((td_target_value - policy_q)**2)
        total_loss+=loss
    
    total_loss = torch.cat(total_loss, dim=0).mean()
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    policy_model.eval()
    policy_model = policy_model.to('cpu')
    target_model = target_model.to('cpu')

    return total_loss.item()