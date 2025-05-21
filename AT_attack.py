#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 11:19:27 2025

@author: joseaguilar
"""

import numpy as np
from torch import nn
import torch


class AdaptiveTargetAttack:
    """First stage: Adaptive Target attack implementation."""
    @torch.no_grad()
    def __init__(self, env, target_policy=None, attack_budget=1.0, attack_magnitude=0.1):
        self.env = env
        self.target_policy = target_policy  # The policy pi_dagger we want to promote
        self.attack_budget = attack_budget  # Total attack budget
        self.attack_magnitude = attack_magnitude  # Maximum magnitude of any single attack
        self.remaining_budget = attack_budget
        self.total_steps = 0
        
    def select_target_action(self, state):
        """Use the target policy pi^dagger to select an action."""
        if self.target_policy is None:
            # If no specific target policy provided, use a simple heuristic
            # This should be replaced with your specific target policy
            # For now, we'll just choose a random action
            return self.env.action_space.sample()
        # Obtain action
        # state = torch.tensor(np.array([state]))
        # Always mean as we do not train target_policy: no need for exploration
        return self.target_policy(state).mean.squeeze()
    
    def compute_attack(self, state, agent_action):
        """
        Compute the attack reward perturbation using the adaptive target strategy.
        Based on "Black-box targeted reward poisoning attack against online deep 
        reinforcement learning" by Xu and Singh.
        """
        if self.remaining_budget <= 0:
            return torch.Tensor([0.0]).squeeze()
            
        target_action = self.select_target_action(state)
        
        # If agent's action matches target, no need to attack
        if agent_action.argmax() == target_action.argmax():
            return torch.Tensor([0.0]).squeeze()
            
        # Compute attack
        # Values per case: similar: 1.; orth: 0., different: -1.
        dist_measure = nn.CosineSimilarity(dim=0)
        attack_value = min(self.attack_magnitude, self.remaining_budget)
        # Similar vectors -> 0.0, different vectors -> 1.0
        # Note that this is negative already!
        dist = (dist_measure(agent_action, target_action) - 1.) / 2.
        # Attack according to paper
        attack_delta = attack_value * dist
        # Determine attack magnitude - promote target action by adding positive reward perturbation
        # This is a simplified version of the adaptive target attack
        self.remaining_budget -= abs(attack_delta)
        self.total_steps += 1
        
        return attack_delta
    
    def reset(self, reset_budget=False):
        """Reset attack parameters for a new episode."""
        # Budget remains depleted across episodes in this implementation
        # Modify this if you want to replenish budget per episode
        if reset_budget:
            self.remaining_budget = self.attack_budget
        
        
        
        
        