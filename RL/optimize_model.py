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
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RL.utils import get_device
from RL.loss import *


def calu_entropy(out_policy, states):
    entropy = 0
    states_2 = states[:,2,:,:].flatten(1)
    for tmp_policy, stmp_2 in zip(out_policy, states_2):
        tmp_p = tmp_policy[stmp_2!=0]
        if len(tmp_p)==0:
            continue
        entropy+=Categorical(logits=tmp_p).entropy()
    return entropy

def optimize_a2cmodel(model, transactions, optimizer, gamma=.9, lam=.6, step=5000):
    device = get_device()
    model = model.train()
    model = model.to(device)

    states = []
    actions = []
    td_target_values = []
    td_target_returns = []

    for transaction in transactions:
        tmp_states = copy.deepcopy(transaction['states'])
        tmp_actions = copy.deepcopy(transaction["actions"])
        tmp_values = copy.deepcopy(transaction["values"])
        tmp_rewards = copy.deepcopy(transaction["rewards"])

        tmp_out = model(tmp_states[:-1].to(device))
        target_value = tmp_out['value'].detach().to('cpu')
        target_return = tmp_out['return'].detach().to('cpu')

        tmp_td_target_value = calu_td_target_value_a2c(target_value, tmp_values, gamma, lam)
        tmp_td_target_return = calu_td_target_value_a2c(target_return, tmp_rewards, gamma, lam)

        states.append(tmp_states[1:])
        actions.append(tmp_actions)
        td_target_values.append(tmp_td_target_value.detach())
        td_target_returns.append(tmp_td_target_return.detach())
    
    states = torch.cat(states, dim=0).to(device)
    actions = torch.cat(actions, dim=0).to(device).to(torch.int64).unsqueeze(1)
    td_target_values = torch.cat(td_target_values, dim=0).to(device)
    td_target_returns = torch.cat(td_target_returns, dim=0).to(device)

    for i in range(2):
        losses = {}
        out = model(states)
        out_policy = Categorical(logits=out['policy'])
        out_values = out['value']
        out_returns = out['return']

        values_advantage = td_target_values - out_values
        returns_advantage = td_target_returns - out_returns
        
        log_probs = out_policy.logits.gather(1, actions)
        advantages = 3*values_advantage + returns_advantage

        losses["actor_loss"] = -(log_probs * advantages.detach()).squeeze(1)
        losses["critic_value_loss"] = F.mse_loss(out_values, td_target_values, reduction='none').squeeze(1)
        losses["critic_return_loss"] = F.smooth_l1_loss(out_returns, td_target_returns, reduction='none').squeeze(1)
        losses['entropy'] = -calu_entropy(out['policy'], states)
        
        loss = 0
        coef = {'actor_loss':1, 'critic_value_loss':1, 'critic_return_loss':1, 'entropy':0.8}
        for key, loss_value in losses.items():
            c = coef[key]
            if key=='entropy':
                c = c**step 
            loss += c*(loss_value).mean()
        
        loss.backward()
        if i==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.first_step(zero_grad=True)
        if i==1:
            # second forward-backward pass
            optimizer.second_step(zero_grad=True)
    model = model.eval()
    model = model.to('cpu')

    return losses["actor_loss"].mean().item(), losses["critic_value_loss"].mean().item(), losses["critic_return_loss"].mean().item()

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