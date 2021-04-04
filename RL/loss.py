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

def calu_td_target_value_a2c(value, rewards, gamma=.9, lam=.6):
    td_target_value = deque()
    td_target_value.appendleft(rewards[0]+gamma*value[0])
    for v, r in zip(value[1:], rewards[1:]):
        a = r+gamma*((1-lam)*v+lam*td_target_value[0])
        td_target_value.appendleft(a)
    td_target_value = torch.cat(list(td_target_value)[::-1]).unsqueeze(1)
    return td_target_value

def calu_td_target_value_dqn(value, rewards, gamma=.9, lam=.6):
    td_target_value = deque()
    max_value = value.max(1)[0].unsqueeze(1)
    td_target_value.appendleft(rewards[0]+gamma*max_value[0])
    for v, r in zip(max_value[1:], rewards[1:]):
        a = r+gamma*((1-lam)*v+lam*td_target_value[0])
        td_target_value.appendleft(a)
    td_target_value = torch.cat(list(td_target_value)[::-1]).unsqueeze(1)
    return td_target_value

def calu_loss(policy, actions, value, td_target_value, states):
    log_probs = policy.logits.gather(1, actions.to(torch.int64).unsqueeze(1))
    #advantages = expected_state_action_values - value
    advantage = td_target_value-value

    policy_loss = -log_probs*(advantage.detach())
    value_loss = advantage**2
    entropy = policy.entropy().unsqueeze(1)
    othello_loss = calu_othello_loss(policy, states)

    return policy_loss, value_loss, entropy, othello_loss

def calu_td_value_loss(policy_values, target_values):
    p = policy_values.squeeze()[1:]
    loss = (1/2)*(target_values-p)**2
    loss = loss.mean()
    return loss

def calu_othello_loss(policy, states):
    p = policy.probs
    _states = copy.deepcopy(states[1:,2,:,:]).flatten(1)
    loss = p[_states==0].mean()
    loss = 0 if torch.isnan(loss).all() else loss
    return loss