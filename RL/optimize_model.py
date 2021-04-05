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
from RL.utils import get_device
from RL.loss import *

def optimize_a2cmodel(model, transactions, optimizer, gamma=.9, lam=.6):
    device = get_device()
    model = model.to(device)

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
    device = get_device()
    policy_model = policy_model.to(device)
    
    target_value = []
    states_for_policy = []
    actions_for_policy = []

    for transaction in transactions:
        states = copy.deepcopy(transaction['states'])
        rewards = copy.deepcopy(transaction["rewards"])
        actions = copy.deepcopy(transaction["actions"])
        
        target_q = target_model(states[:-1])['policy'].detach()
        td_target_value = calu_td_target_value_dqn(target_q, rewards, gamma, lam)
        
        states_for_policy.append(states[1:])
        actions_for_policy.append(actions.unsqueeze(1))
        target_value.append(td_target_value.detach())
    
    target_values = torch.cat(target_value, dim=0).to(device)
    states_for_policy = torch.cat(states_for_policy, dim=0).to(device)
    actions_for_policy = torch.cat(actions_for_policy, dim=0).to(device)
    
    #print(f"target_values.size()  {target_values.size()}")
    #print(f"states_for_policy.size()  {states_for_policy.size()}")
    #print(f"actions_for_policy.size()  {actions_for_policy.size()}")

    policy_q = policy_model(states_for_policy)['policy']
    policy_q = policy_q.gather(1, actions_for_policy.to(torch.int64))
    
    loss = nn.MSELoss()(policy_q, target_values)

    sam=True
    if not sam:
        optimizer.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1)
        optimizer.step()
        
    else:
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1)
        optimizer.first_step(zero_grad=True)
        policy_q = policy_model(states_for_policy)['policy']
        policy_q = policy_q.gather(1, actions_for_policy.to(torch.int64))
        nn.MSELoss()(policy_q, target_values).backward()
        optimizer.second_step(zero_grad=True)
    
    policy_model.eval()
    policy_model = policy_model.to('cpu')
    target_model = target_model.to('cpu')

    return loss.item()