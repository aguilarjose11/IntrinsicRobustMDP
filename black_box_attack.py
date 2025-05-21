#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 09:41:50 2025

@author: joseaguilar
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time
from typing import Dict, List, Tuple, Optional, Any

class ReplayBuffer:
    """Simple replay buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.memory)

class LinearMDPModel:
    """Model representing the components of a linear MDP: φ(s,a), μ(s), and θ."""
    
    def __init__(self, state_dim, action_dim, feature_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Initialize networks for φ(s,a), μ(s), and weights θ
        self.phi_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        self.mu_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
        self.theta = nn.Parameter(torch.randn(feature_dim))
        
    def phi(self, state, action):
        """Compute φ(s,a) - the feature representation of state-action pairs."""
        if isinstance(action, int) or (isinstance(action, np.ndarray) and action.shape == ()):
            # Convert scalar action to one-hot if needed
            action_vec = np.zeros(self.action_dim)
            action_vec[action] = 1
            action = action_vec
            
        sa_pair = np.concatenate([state, action])
        return self.phi_network(torch.FloatTensor(sa_pair))
    
    def mu(self, state):
        """Compute μ(s) - the feature representation of state."""
        return self.mu_network(torch.FloatTensor(state))
    
    def compute_q_value(self, state, action, w):
        """Compute Q(s,a) = <φ(s,a), w> for a given w."""
        phi_sa = self.phi(state, action)
        return torch.dot(phi_sa, w)
    
    def compute_reward(self, state, action):
        """Compute reward r(s,a) = <φ(s,a), θ>."""
        phi_sa = self.phi(state, action)
        return torch.dot(phi_sa, self.theta)

class CTRLLCBTrainer:
    """Implementation of CTRL-LCB algorithm to learn a linear MDP representation."""
    
    def __init__(self, state_dim, action_dim, feature_dim, lr=1e-3):
        self.linear_mdp = LinearMDPModel(state_dim, action_dim, feature_dim)
        self.optimizer = optim.Adam(self.linear_mdp.parameters(), lr=lr)
        self.feature_dim = feature_dim
        
    def contrastive_loss(self, anchor, positive, negative, margin=1.0):
        """Compute contrastive loss for representation learning."""
        pos_distance = torch.norm(anchor - positive, dim=1)
        neg_distance = torch.norm(anchor - negative, dim=1)
        loss = torch.relu(pos_distance - neg_distance + margin).mean()
        return loss
    
    def train_step(self, batch_data):
        """Perform one training step on a batch of data."""
        states, actions, rewards, next_states, dones = batch_data
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)
        
        # Get representations
        phi_sa_list = []
        for i in range(len(states)):
            phi_sa = self.linear_mdp.phi(states[i], actions[i])
            phi_sa_list.append(phi_sa)
        phi_sa_batch = torch.stack(phi_sa_list)
        
        # Compute predicted rewards
        pred_rewards = torch.sum(phi_sa_batch * self.linear_mdp.theta, dim=1)
        reward_loss = nn.MSELoss()(pred_rewards, rewards_tensor)
        
        # Compute contrastive loss for representation learning
        # Create positive and negative samples
        # For simplicity, using random other states as negatives
        indices = torch.randperm(len(states))
        neg_states = states[indices]
        neg_actions = actions[indices]
        
        anchor_phi = phi_sa_batch
        positive_phi = []
        for i in range(len(states)):
            # Use next state and a random action as positive example
            # In a real implementation, this would use the policy to select action
            rand_action = np.random.randint(0, self.linear_mdp.action_dim)
            pos_phi = self.linear_mdp.phi(next_states[i], rand_action)
            positive_phi.append(pos_phi)
        positive_phi = torch.stack(positive_phi)
        
        negative_phi = []
        for i in range(len(states)):
            neg_phi = self.linear_mdp.phi(neg_states[i], neg_actions[i])
            negative_phi.append(neg_phi)
        negative_phi = torch.stack(negative_phi)
        
        c_loss = self.contrastive_loss(anchor_phi, positive_phi, negative_phi)
        
        # Total loss
        total_loss = reward_loss + c_loss
        
        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), reward_loss.item(), c_loss.item()
    
    def train(self, replay_buffer, epochs=100, batch_size=32):
        """Train the linear MDP model using collected data."""
        losses = []
        for epoch in range(epochs):
            if len(replay_buffer) < batch_size:
                continue
                
            batch_data = replay_buffer.sample()
            loss, r_loss, c_loss = self.train_step(batch_data)
            losses.append(loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Reward Loss = {r_loss:.4f}, Contrastive Loss = {c_loss:.4f}")
                
        return self.linear_mdp, losses

class AdaptiveTargetAttack:
    """First stage: Adaptive Target attack implementation."""
    
    def __init__(self, env, target_policy=None, attack_budget=1.0, attack_magnitude=0.1):
        self.env = env
        self.target_policy = target_policy  # The policy π† we want to promote
        self.attack_budget = attack_budget  # Total attack budget
        self.attack_magnitude = attack_magnitude  # Maximum magnitude of any single attack
        self.remaining_budget = attack_budget
        self.total_steps = 0
        
    def select_target_action(self, state):
        """Use the target policy π† to select an action."""
        if self.target_policy is None:
            # If no specific target policy provided, use a simple heuristic
            # This should be replaced with your specific target policy
            # For now, we'll just choose a random action
            return self.env.action_space.sample()
        return self.target_policy(state)
    
    def compute_attack(self, state, agent_action):
        """
        Compute the attack reward perturbation using the adaptive target strategy.
        Based on "Black-box targeted reward poisoning attack against online deep 
        reinforcement learning" by Xu and Singh.
        """
        if self.remaining_budget <= 0:
            return 0.0
            
        target_action = self.select_target_action(state)
        
        # If agent's action matches target, no need to attack
        if agent_action == target_action:
            return 0.0
            
        # Determine attack magnitude - promote target action by adding positive reward perturbation
        # This is a simplified version of the adaptive target attack
        attack_value = min(self.attack_magnitude, self.remaining_budget)
        self.remaining_budget -= attack_value
        self.total_steps += 1
        
        return attack_value
    
    def reset(self):
        """Reset attack parameters for a new episode."""
        # Budget remains depleted across episodes in this implementation
        # Modify this if you want to replenish budget per episode
        pass

class LinearMDPAttackabilityChecker:
    """Determine attackability of a linear MDP using collected data."""
    
    def __init__(self, state_dim, action_dim, feature_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
    def solve_for_attackability(self, linear_mdp_model, data, target_policy):
        """
        Solve for w† and ε to determine attackability.
        Returns:
            - is_attackable: bool - whether the MDP is attackable
            - w_dagger: numpy array - the attack parameter if attackable
            - epsilon: float - the gap value
        """
        # Extract states and actions from data
        states = [item[0] for item in data]
        
        # Construct optimization problem to find w† that maximizes the gap
        # This is a simplified version - in practice, you'd use a proper optimization solver
        
        # Initialize with random w
        w_dagger = np.random.randn(self.feature_dim)
        best_epsilon = float('-inf')
        
        # Simple gradient ascent to find w† (in practice use a proper solver)
        for _ in range(1000):  # Number of optimization steps
            # Compute gradient of epsilon with respect to w
            grad = np.zeros(self.feature_dim)
            current_epsilon = float('inf')
            
            for state in states:
                # Get target action from target policy
                target_action = target_policy(state)
                
                # Compute φ(s,a†)
                phi_s_target = linear_mdp_model.phi(state, target_action).detach().numpy()
                
                # Compute the minimum gap over all other actions
                min_gap = float('inf')
                for action in range(self.action_dim):
                    if action == target_action:
                        continue
                        
                    # Compute φ(s,a)
                    phi_s_a = linear_mdp_model.phi(state, action).detach().numpy()
                    
                    # Compute gap: Q(s,a†) - Q(s,a) = <φ(s,a†) - φ(s,a), w>
                    gap = np.dot(phi_s_target - phi_s_a, w_dagger)
                    min_gap = min(min_gap, gap)
                    
                # Update minimum epsilon across all states
                current_epsilon = min(current_epsilon, min_gap)
                
                # Update gradient (simplified)
                for action in range(self.action_dim):
                    if action == target_action:
                        continue
                    phi_s_a = linear_mdp_model.phi(state, action).detach().numpy()
                    gap = np.dot(phi_s_target - phi_s_a, w_dagger)
                    
                    # Add to gradient if this is the action achieving the minimum gap
                    if abs(gap - min_gap) < 1e-6:
                        grad += (phi_s_target - phi_s_a)
            
            # Update w† using gradient ascent
            learning_rate = 0.01
            w_dagger += learning_rate * grad
            w_dagger = w_dagger / np.linalg.norm(w_dagger)  # Normalize
            
            # Update best epsilon
            if current_epsilon > best_epsilon:
                best_epsilon = current_epsilon
                best_w_dagger = w_dagger.copy()
        
        # Determine attackability
        is_attackable = best_epsilon > 0
        
        return is_attackable, best_w_dagger, best_epsilon

class SecondStageAttack:
    """
    Second-stage attack using the computed w† to create targeted reward perturbations.
    This attack directly modifies rewards based on the Q-function gap.
    """
    
    def __init__(self, linear_mdp_model, w_dagger, target_policy, attack_budget=1.0, attack_magnitude=0.1):
        self.linear_mdp_model = linear_mdp_model
        self.w_dagger = torch.FloatTensor(w_dagger)
        self.target_policy = target_policy
        self.attack_budget = attack_budget
        self.attack_magnitude = attack_magnitude
        self.remaining_budget = attack_budget
        self.total_steps = 0
        
    def compute_attack(self, state, agent_action):
        """Compute the attack reward perturbation using w†."""
        if self.remaining_budget <= 0:
            return 0.0
            
        target_action = self.target_policy(state)
        
        # If agent's action matches target, no need to attack
        if agent_action == target_action:
            return 0.0
            
        # Compute Q-values using w†
        q_target = self.linear_mdp_model.compute_q_value(state, target_action, self.w_dagger).item()
        q_agent = self.linear_mdp_model.compute_q_value(state, agent_action, self.w_dagger).item()
        
        # Determine attack magnitude based on Q-value difference
        q_diff = q_target - q_agent
        
        # If target action has higher Q-value, no need to attack
        if q_diff >= 0:
            return 0.0
            
        # Otherwise, add perturbation proportional to Q-value difference
        attack_value = min(self.attack_magnitude, abs(q_diff), self.remaining_budget)
        self.remaining_budget -= attack_value
        self.total_steps += 1
        
        return attack_value
    
    def reset(self):
        """Reset attack parameters for a new episode."""
        # Budget remains depleted across episodes
        pass

class BlackBoxAttackExperiment:
    """
    Main experiment class that coordinates the two-stage attack.
    """
    
    def __init__(self, env_name, feature_dim=10, first_stage_budget=10.0, 
                 second_stage_budget=20.0, attack_magnitude=0.1, 
                 first_stage_episodes=100, max_total_episodes=200):
        
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        
        # Handle both discrete and continuous action spaces
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:  # Continuous
            self.action_dim = self.env.action_space.shape[0]
            self.is_discrete = False
        
        self.feature_dim = feature_dim
        self.first_stage_budget = first_stage_budget
        self.second_stage_budget = second_stage_budget
        self.attack_magnitude = attack_magnitude
        self.first_stage_episodes = first_stage_episodes
        self.max_total_episodes = max_total_episodes
        
        # Initialize replay buffer for data collection
        self.replay_buffer = ReplayBuffer(buffer_size=100000, batch_size=64)
        
        # Initialize victim agent - this should be replaced with your actual RL agent
        # For example, a DQN or PPO implementation
        self.victim_agent = self._create_dummy_agent()
        
        # Define target policy (π†) - replace with your actual target policy
        self.target_policy = self._create_target_policy()
        
    def _create_dummy_agent(self):
        """Create a simple placeholder agent for demonstration."""
        def agent_policy(state):
            # Random policy for demonstration purposes
            # Replace with your actual victim agent policy
            return self.env.action_space.sample()
        return agent_policy
    
    def _create_target_policy(self):
        """Create a simple target policy for demonstration."""
        def target_policy(state):
            # For discrete actions: always choose action 0 as the target
            # For continuous: choose action in a specific direction
            if self.is_discrete:
                return 0
            else:
                # Return a specific continuous action
                return np.zeros(self.action_dim)
        return target_policy
    
    def run_first_stage(self):
        """
        Run first stage of the attack using Adaptive Target strategy 
        and collect data for linear MDP estimation.
        """
        print("Starting First Stage: Data Collection with Adaptive Target Attack")
        
        # Initialize first-stage attack
        attack = AdaptiveTargetAttack(
            self.env, 
            target_policy=self.target_policy,
            attack_budget=self.first_stage_budget,
            attack_magnitude=self.attack_magnitude
        )
        
        # Collect data while attacking
        collected_data = []
        episode_rewards = []
        
        for episode in range(self.first_stage_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            truncated = False
            attack.reset()
            
            while not (done or truncated):
                # Agent selects action
                action = self.victim_agent(state)
                
                # Compute attack
                attack_value = attack.compute_attack(state, action)
                
                # Execute action in environment
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Apply attack to reward
                modified_reward = reward + attack_value
                
                # Store experience in replay buffer
                self.replay_buffer.add(state, action, modified_reward, next_state, done)
                
                # Also store for later analysis
                collected_data.append((state, action, modified_reward, next_state, done))
                
                # Update state and episode reward
                state = next_state
                episode_reward += modified_reward
            
            episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Remaining Budget = {attack.remaining_budget:.2f}")
        
        print(f"First Stage Complete. Collected {len(collected_data)} transitions.")
        return collected_data, episode_rewards
    
    def train_linear_mdp(self, collected_data):
        """Train linear MDP model using CTRL-LCB algorithm on collected data."""
        print("Training Linear MDP Model")
        
        # Initialize CTRL-LCB trainer
        trainer = CTRLLCBTrainer(self.state_dim, self.action_dim, self.feature_dim)
        
        # Train the model
        linear_mdp_model, losses = trainer.train(self.replay_buffer, epochs=500)
        
        print(f"Linear MDP Training Complete. Final Loss: {losses[-1]:.4f}")
        return linear_mdp_model
    
    def check_attackability(self, linear_mdp_model, collected_data):
        """Determine if the environment is attackable."""
        print("Checking Attackability")
        
        checker = LinearMDPAttackabilityChecker(self.state_dim, self.action_dim, self.feature_dim)
        is_attackable, w_dagger, epsilon = checker.solve_for_attackability(
            linear_mdp_model, collected_data, self.target_policy
        )
        
        print(f"Attackability Check Complete. Is Attackable: {is_attackable}, Epsilon: {epsilon:.4f}")
        return is_attackable, w_dagger, epsilon
    
    def run_second_stage(self, linear_mdp_model, w_dagger):
        """Run second stage of the attack using the computed w†."""
        print("Starting Second Stage: Attacking with Computed Parameters")
        
        # Initialize second-stage attack
        attack = SecondStageAttack(
            linear_mdp_model,
            w_dagger,
            self.target_policy,
            attack_budget=self.second_stage_budget,
            attack_magnitude=self.attack_magnitude
        )
        
        # Run episodes with the second-stage attack
        remaining_episodes = self.max_total_episodes - self.first_stage_episodes
        episode_rewards = []
        target_action_freq = []  # Track frequency of target action selection
        
        for episode in range(remaining_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            truncated = False
            attack.reset()
            target_actions = 0
            total_actions = 0
            
            while not (done or truncated):
                # Agent selects action
                action = self.victim_agent(state)
                
                # Check if action matches target
                if (self.is_discrete and action == self.target_policy(state)) or \
                   (not self.is_discrete and np.allclose(action, self.target_policy(state))):
                    target_actions += 1
                total_actions += 1
                
                # Compute attack
                attack_value = attack.compute_attack(state, action)
                
                # Execute action in environment
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Apply attack to reward
                modified_reward = reward + attack_value
                
                # Update state and episode reward
                state = next_state
                episode_reward += modified_reward
            
            episode_rewards.append(episode_reward)
            target_action_freq.append(target_actions / max(1, total_actions))
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                      f"Target Action Frequency = {target_action_freq[-1]:.2f}, "
                      f"Remaining Budget = {attack.remaining_budget:.2f}")
        
        print(f"Second Stage Complete. Average Target Action Frequency: {np.mean(target_action_freq):.2f}")
        return episode_rewards, target_action_freq
    
    def run_experiment(self):
        """Run the complete two-stage black-box attack experiment."""
        print(f"Starting Black-Box Attack Experiment on {self.env}")
        start_time = time.time()
        
        # Run first stage
        collected_data, first_stage_rewards = self.run_first_stage()
        
        # Train linear MDP model
        linear_mdp_model = self.train_linear_mdp(collected_data)
        
        # Check attackability
        is_attackable, w_dagger, epsilon = self.check_attackability(linear_mdp_model, collected_data)
        
        # If attackable, run second stage
        second_stage_rewards = []
        target_action_freq = []
        if is_attackable:
            second_stage_rewards, target_action_freq = self.run_second_stage(linear_mdp_model, w_dagger)
        else:
            print("Environment is not attackable. Stopping experiment.")
        
        # Compute and report results
        experiment_duration = time.time() - start_time
        results = {
            "env_name": self.env.unwrapped.spec.id,
            "is_attackable": is_attackable,
            "epsilon": epsilon,
            "first_stage_rewards": first_stage_rewards,
            "second_stage_rewards": second_stage_rewards,
            "target_action_freq": target_action_freq,
            "experiment_duration": experiment_duration
        }
        
        print(f"Experiment Complete in {experiment_duration:.2f} seconds.")
        return results

if __name__ == "__main__":
    # Example usage
    env_name = "CartPole-v1"  # Replace with your target environment
    experiment = BlackBoxAttackExperiment(
        env_name=env_name,
        feature_dim=20,
        first_stage_budget=50.0,
        second_stage_budget=100.0,
        first_stage_episodes=50,
        max_total_episodes=100
    )
    
    results = experiment.run_experiment()
    
    # Print summary of results
    print("\nExperiment Results Summary:")
    print(f"Environment: {results['env_name']}")
    print(f"Attackable: {results['is_attackable']}")
    print(f"Epsilon (Gap): {results['epsilon']:.4f}")
    print(f"Average First Stage Reward: {np.mean(results['first_stage_rewards']):.2f}")
    
    if results['is_attackable']:
        print(f"Average Second Stage Reward: {np.mean(results['second_stage_rewards']):.2f}")
        print(f"Final Target Action Frequency: {results['target_action_freq'][-1]:.2f}")