#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:23:10 2025

@author: joseaguilar
"""

import torch
from torch import nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


from AT_attack import AdaptiveTargetAttack
from CTRL_networks import DiagGaussianActor
from rl_algorithms import REINFORCE
from Characterize_gym import DiscreteActionWrapper, DiscreteObservationWrapper, make_discrete_env
from CTRL_wrapper import CTRLMDPWrapper


def _get_flattened_dim(space: spaces.Space) -> int:
    """Calculate the flattened dimension of a space."""
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    elif isinstance(space, spaces.Discrete):
        return space.n
    else:
        raise TypeError(f"Cannot get flattened dim for space type {type(space)}")
        
        
def _preprocess_obs(self, obs: np.ndarray) -> torch.Tensor:
    """Preprocess observation (flatten and convert to tensor)."""
    flat_obs = obs.reshape(-1)
    return torch.tensor(flat_obs, dtype=torch.float32, device=self.device).unsqueeze(0)


def calculate_model_norm(model1, model2, norm_type=2):
    """
    Calculates the norm between the parameters of two PyTorch models.

    Args:
        model1: The first PyTorch model.
        model2: The second PyTorch model.
        norm_type: The type of norm to calculate (e.g., 1 for L1, 2 for L2).

    Returns:
        The norm between the models' parameters.
    """
    dist_measure = nn.CosineSimilarity(dim=0)
    params1 = torch.cat([p.data.flatten() for p in model1.parameters()])
    params2 = torch.cat([p.data.flatten() for p in model2.parameters()])
    return dist_measure(params1, params2)
    # return torch.norm(params1 - params2, norm_type)


if __name__ == "__main__":
    print('Start of test. Constructing environment.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tested: CartPole-v1
    feature_dim = 1_024
    env = gym.make("CartPole-v1")
    env = CTRLMDPWrapper(env,
                        feature_dim=feature_dim,
                        hidden_dim=512,
                        learning_rate=5e-5, # Adjusted LR might be needed
                        batch_size=512,
                        buffer_size=100_000,
                        train_freq=5,
                        gradient_steps=1,
                        learning_starts=(learning_starts:=25_000),
                        tau=0.001,
                        dynamics_loss_weight=0.7, # Can tune these weights
                        reward_loss_weight=0.9,   # Example: down-weight reward loss
                        ctrl_gamma=(gamma:=0.99),
                        device="auto")
    
    # HalfCheetah: Tested
    # env = gym.make("HalfCheetah-v5")
    # env = DiscreteActionWrapper(env, n_bins_per_dim=5)
    print("Env constructed. Constructing target policy...")
    
    state_dim = _get_flattened_dim(env.observation_space)
    
    print(f"Observation Dim: {state_dim}, Action Dim: {env.action_space.n}")
    
    actor = DiagGaussianActor(obs_dim=state_dim, 
                              action_dim=env.action_space.n, 
                              hidden_dim=512, 
                              hidden_depth=2, 
                              log_std_bounds=[-5., 2.],).to(device)
    
    print("Target policy Constructed. Constructing AT attack...")
    
    at_attack = AdaptiveTargetAttack(env=env, target_policy=actor, attack_budget=100., attack_magnitude=1.)
    
    print("AT attack Constructed. Constructing Victim Algorithm/Agent...")
    
    num_episodes = 1_000
    total_steps = 0
    print_interval = 10
    log_rewards = {
        'episode': [], 
        'reward': [], 
        'perturbation': [], 
        'budget': [],
        'policy difference': []
    }
    cummulative_rewards = 0. # ish: every so few intervals
    cummulative_perturbation = 0. # ibid
    total_budget_used = 0.
    
    # Test DRL algorithm
    agent = REINFORCE(state_dim, env.action_space.n.item(), device)
    
    print('Victim algorithm constructed. Starting training.')
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = torch.Tensor(obs).to(device)
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            action = agent.sample_action(obs)
            # action = linear_mdp_env.actor(linear_mdp_env._preprocess_obs(obs)).rsample().argmax().cpu().numpy()
            # action = linear_mdp_env.action_space.sample()
            # Use  .mean.argmax().cpu().numpy() instead of rsample for mean
            next_obs, reward, terminated, truncated, info = env.step(np.argmax(action.detach().cpu().numpy()))
            done = terminated or truncated
            # Poison with AT
            with torch.no_grad():
                perturbation = at_attack.compute_attack(obs, action).item()
                reward_ = reward + perturbation
            agent.rewards.append(reward_)
            obs = torch.Tensor(next_obs).to(device)
            # Perturbation is always negative.
            cummulative_perturbation += perturbation
            total_budget_used += abs(perturbation)
            # Other stats
            episode_reward += reward
            cummulative_rewards += reward
            episode_steps += 1
            total_steps += 1
        agent.update()
        at_attack.reset(reset_budget=True)
        if (episode + 1) % print_interval == 0:
            policy_diff = calculate_model_norm(actor, agent.net).item()
            print(f"Episode: {episode+1}/{num_episodes}, Steps: {episode_steps}, Total Steps: {total_steps}, Reward: {cummulative_rewards:.2f}/{cummulative_rewards + cummulative_perturbation:.2f}, Attack: {cummulative_perturbation:.3f}, Difference: {policy_diff:.3}")
            log_rewards['episode'].append(episode)
            log_rewards['reward'].append(cummulative_rewards)
            # Reset afterwards
            log_rewards['perturbation'].append(cummulative_perturbation)
            # Will keep growing until convergence (attack success)
            log_rewards['budget'].append(total_budget_used)
            log_rewards['policy difference'].append(policy_diff)
            
            # Reset cummulative values
            cummulative_rewards = 0.
            cummulative_perturbation = 0.

    print("\n--- Training Finished (V2 Wrapper) ---")
    log_pd = pd.melt(pd.DataFrame(log_rewards), id_vars=['episode'], value_vars=['reward', 'perturbation', 'budget', 'policy difference'])
    g = sns.FacetGrid(log_pd, col="variable", sharex=False, sharey=False)
    g.map(sns.lineplot, 'episode', 'value')
    g.add_legend()
    plt.show()