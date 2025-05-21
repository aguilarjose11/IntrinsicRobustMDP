#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:22:31 2025

@author: joseaguilar
"""

from CTRL_wrapper import CTRLMDPWrapper
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

from attackability import calculate_M_pi_dagger, characterize_robustness_torch
from Characterize_gym import DiscreteActionWrapper, DiscreteObservationWrapper, make_discrete_env




if __name__ == "__main__":
    print("--- Testing V2 Wrapper with CartPole-v1 ---")
    base_env = gym.make("CartPole-v1")
    # base_env = make_discrete_env("CartPole-v1", action_bins=0.1, obs_bins=100)
    
    feature_dim = 1_024

    # Use the modified wrapper V2
    linear_mdp_env = CTRLMDPWrapper(
        base_env,
        feature_dim=feature_dim,
        hidden_dim=512,
        learning_rate=5e-5, # Adjusted LR might be needed
        batch_size=512,
        buffer_size=100_000,
        train_freq=5,
        gradient_steps=1,
        learning_starts=(learning_starts:=60_000),
        tau=0.001,
        dynamics_loss_weight=0.7, # Can tune these weights
        reward_loss_weight=0.9,   # Example: down-weight reward loss
        ctrl_gamma=(gamma:=0.99),
        device="auto"
    )

    # --- Interaction loop (same as before) ---
    num_episodes = 8_000
    # num_steps = 40_500
    total_steps = 0
    print_interval = 50
    # policy_start_ep = 2_500 
    
    log_rewards = {'episode': [], 'reward': []}
    cummulative_rewards = 0.

    for episode in range(num_episodes):
        obs, info = linear_mdp_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            with torch.no_grad():
                if total_steps > learning_starts and np.random.uniform(0, 1) >= 0.05:
                    # action = linear_mdp_env.actor(linear_mdp_env._preprocess_obs(obs)).mean.argmax().cpu().numpy()
                    action = linear_mdp_env.actor(linear_mdp_env._preprocess_obs(obs)).rsample().argmax().cpu().numpy()
                else:
                    action = linear_mdp_env.action_space.sample()
                # Use  .mean.argmax().cpu().numpy() instead of rsample for mean
            next_obs, reward, terminated, truncated, info = linear_mdp_env.step(action)
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            cummulative_rewards += reward
            episode_steps += 1
            total_steps += 1
        if (episode + 1) % print_interval == 0:
            print(f"Episode: {episode+1}/{num_episodes}, Steps: {episode_steps}, Total Steps: {total_steps}, Reward: {cummulative_rewards/print_interval:.2f}|{cummulative_rewards:.2f}")
            log_rewards['episode'].append(episode)
            log_rewards['reward'].append(cummulative_rewards)
            cummulative_rewards = 0.

    print("\n--- Training Finished (V2 Wrapper) ---")
    sns.lineplot(data=log_rewards, x='episode', y='reward')

    # --- Access learned parameters ---
    theta = linear_mdp_env.get_reward_params()
    M = linear_mdp_env.get_transition_params()

    print(f"Learned theta (reward weights) shape: {theta.shape}")
    # print(f"Theta sample: {theta[:5]}...")
    print(f"Learned M (transition matrix) shape: {M.shape}")
    # print(f"M sample (top-left 5x5):\n{M[:5, :5]}")

    # Verify components access
    components = linear_mdp_env.get_learned_components()
    print("Accessing components directly:")
    print("  phi:", components['phi'])
    print("  M_matrix_layer (represents M^T):", components['M_matrix_layer'])
    print("  mu:", components['mu'])
    print("  reward_head (represents theta^T):", components['reward_head'])


    # Characterization
    n_samples = 10_000
    print(f"Testing Characterization on {n_samples} samples (x2 for both constraints.)")
    
    phi_features = components['phi']
    theta_star = linear_mdp_env.get_reward_params()# components['reward_head']
    action_space = list(range(linear_mdp_env.action_space.n)) # or self._get_flattened_dim(self.action_space) # see CTRL_wrapper
    # Relevant states sampled from pi_target
    pi_target = lambda s: linear_mdp_env.action_space.sample()    
    states, _, _, _, _ = linear_mdp_env.replay_buffer.sample(n_samples, device='cpu')
    # batch x state_dim
    relevant_states = torch.unique(states, dim=0)
    
    # We can also obtain M_pi_dagger as:
    # M_pi_dagger = calculate_M_pi_dagger(phi_features, pi_target, relevant_states, gamma, action_space, )
    M_pi_dagger = linear_mdp_env.get_transition_params() # components['M_matrix_layer']
    # Consider using calculate_M_pi_dagger to use LSTD
    action_dim = linear_mdp_env.action_dim
    
    epsilon, theta_dagger, status = characterize_robustness_torch(phi_features, 
                                                                  pi_target, 
                                                                  theta_star, 
                                                                  relevant_states, 
                                                                  action_space, 
                                                                  M_pi_dagger,
                                                                  linear_mdp_env.d,
                                                                  action_dim,
                                                                  solver='SCS', # SCS(y), CVXOPT(?), CLARABEL(x)
                                                                  device='cuda')
    


    linear_mdp_env.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    