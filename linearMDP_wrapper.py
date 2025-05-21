#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:24:26 2025

@author: joseaguilar
"""

import gymnasium as gym
from gymnasium import spaces, Wrapper
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
from typing import Optional, Tuple, Any, Dict, SupportsFloat

# --- ReplayBuffer and PhiNetwork remain the same ---
# ... (Include ReplayBuffer and PhiNetwork classes from previous answer) ...
class ReplayBuffer:
    """A simple FIFO experience replay buffer."""
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        """Add a new experience to memory."""
        action_processed = action
        if isinstance(action, int):
             action_processed = np.array([action])
        elif isinstance(action, (np.ndarray, list, tuple)):
             action_processed = np.array(action)

        experience = (state, action_processed, reward, next_state, float(done))
        self.buffer.append(experience)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

class PhiNetwork(nn.Module):
    """Neural network to compute features phi(s, a)."""
    def __init__(self, state_dim: int, action_dim: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dtype != torch.float32:
             action = action.float()
        sa_input = torch.cat([state, action], dim=-1)
        return self.network(sa_input)
# -----------------------------------------------------

class LinearMDPLearnerWrapperV2(Wrapper): # Renamed for clarity
    """
    Gymnasium wrapper that learns a linear MDP representation (phi, M, theta)
    using a CTRL-inspired approach with explicit M and theta.
    """
    def __init__(
        self,
        env: gym.Env,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 1000,
        tau: float = 0.005,
        dynamics_loss_weight: float = 1.0, # Weight for || M^T phi - gamma phi' ||^2
        reward_loss_weight: float = 1.0,   # Weight for || theta^T phi - r ||^2
        # Removed ctrl_lambda, added specific weights
        ctrl_gamma: float = 0.99,          # Discount factor used in dynamics target
        device: str = "auto",
    ):
        super().__init__(env)

        # --- Hyperparameters ---
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.tau = tau
        self.dynamics_loss_weight = dynamics_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.ctrl_gamma = ctrl_gamma

        # --- Device Setup ---
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # --- State and Action Space Handling (same as before) ---
        self._validate_spaces()
        self.state_dim = self._get_flattened_dim(self.observation_space)
        if isinstance(self.action_space, spaces.Discrete):
            self.action_dim = self.action_space.n
            self._action_processor = self._one_hot_encode_action
        elif isinstance(self.action_space, spaces.Box):
            self.action_dim = self._get_flattened_dim(self.action_space)
            self._action_processor = lambda a: np.array(a, dtype=np.float32)
        else:
            raise NotImplementedError(f"Action space {type(self.action_space)} not supported.")

        # --- Learnable Components ---
        # Feature extractor network
        self.phi_network = PhiNetwork(self.state_dim, self.action_dim, feature_dim, hidden_dim).to(self.device)
        self.phi_target_network = PhiNetwork(self.state_dim, self.action_dim, feature_dim, hidden_dim).to(self.device)
        self.phi_target_network.load_state_dict(self.phi_network.state_dict())
        self.phi_target_network.eval()

        # Explicit Transition Matrix (M^T represented by Linear layer weights)
        # Maps phi(s,a) -> predicted next phi(s',a') (before discount)
        self.M_matrix_layer = nn.Linear(feature_dim, feature_dim, bias=False).to(self.device)

        # Explicit Reward Weights (theta^T represented by Linear layer weights)
        # Maps phi(s,a) -> predicted reward
        self.reward_head = nn.Linear(feature_dim, 1, bias=False).to(self.device) # Often bias is excluded

        # --- Optimizer (Optimizes phi, M, and theta) ---
        self.optimizer = optim.Adam(
            list(self.phi_network.parameters()) +
            list(self.M_matrix_layer.parameters()) +
            list(self.reward_head.parameters()),
            lr=learning_rate
        )

        # --- Data Collection & Training State (same as before) ---
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self._step_count = 0
        self._last_obs: Optional[np.ndarray] = None
        self._current_trajectory_actions = []

    # --- Helper methods (_validate_spaces, _get_flattened_dim, _preprocess_obs,
    # --- _one_hot_encode_action, _process_action_for_input) remain the same ---
    # ... (Include these methods from the previous answer) ...
    def _validate_spaces(self):
        """Check if observation and action spaces are supported."""
        if not isinstance(self.observation_space, spaces.Box):
            raise NotImplementedError(f"Observation space {type(self.observation_space)} not supported. Use FlattenObservation wrapper.")
        if not isinstance(self.action_space, (spaces.Discrete, spaces.Box)):
             raise NotImplementedError(f"Action space {type(self.action_space)} not supported.")
        if len(self.observation_space.shape) > 1:
             print("Warning: Observation space has multiple dimensions. Ensure it's flattened if necessary before this wrapper.")

    def _get_flattened_dim(self, space: spaces.Space) -> int:
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

    def _one_hot_encode_action(self, action: int) -> np.ndarray:
        """Convert discrete action to one-hot vector."""
        one_hot = np.zeros(self.action_dim, dtype=np.float32)
        one_hot[action] = 1.0
        return one_hot

    def _process_action_for_input(self, action: torch.Tensor) -> torch.Tensor:
         """ Process batch of actions for network input (e.g. ensure float/one-hot) """
         if isinstance(self.action_space, spaces.Discrete) and action.dtype != torch.float32:
              action = action.long()
              action_one_hot = torch.zeros(action.size(0), self.action_dim, device=self.device)
              action_one_hot.scatter_(1, action, 1)
              return action_one_hot
         elif isinstance(self.action_space, spaces.Box):
              return action.float()
         return action

    # --- Core Methods (get_feature, step, reset) remain largely the same ---
    # ... (Include get_feature, step, reset methods from the previous answer) ...
    #     (Ensure step calls self.train_ctrl())
    @torch.no_grad()
    def get_feature(self, obs: np.ndarray, action: Any) -> np.ndarray:
        """Compute the learned feature vector phi(s, a)."""
        self.phi_network.eval()
        state_tensor = self._preprocess_obs(obs)
        processed_action = self._action_processor(action)
        action_tensor = torch.tensor(processed_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_input = self._process_action_for_input(action_tensor)
        feature_tensor = self.phi_network(state_tensor, action_input)
        self.phi_network.train()
        return feature_tensor.squeeze(0).cpu().numpy()

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Take step, store transition, potentially train."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        done = terminated or truncated

        if self._last_obs is not None:
            processed_action_storage = self._action_processor(action)
            self.replay_buffer.add(self._last_obs, processed_action_storage, reward, obs, done)
            self._current_trajectory_actions.append(processed_action_storage)

        if self._step_count >= self.learning_starts and self._step_count % self.train_freq == 0:
            if len(self.replay_buffer) >= self.batch_size:
                for _ in range(self.gradient_steps):
                    self.train_ctrl()

        self._last_obs = obs
        if done:
           # info['final_observation'] = self._last_obs # Gymnasium standard
           if hasattr(self, '_last_obs') and self._last_obs is not None:
               info['final_observation'] = np.array(self._last_obs, dtype=self.observation_space.dtype) # Ensure correct dtype for Gym standard
           self._current_trajectory_actions = []

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and internal state."""
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._current_trajectory_actions = []
        return obs, info
    # --------------------------------------------------------------------------

    def train_ctrl(self):
        """Sample batch and perform gradient step with explicit M and theta."""
        self.phi_network.train()
        self.M_matrix_layer.train()
        self.reward_head.train()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)
        actions_input = self._process_action_for_input(actions)

        # *** Placeholder for next actions a' ***
        # As before, this requires a strategy (e.g., sample from policy, mean action).
        # Using the same action as a placeholder:
        next_actions_input = actions_input # Simplification!

        # --- Compute Loss ---
        loss = self._compute_explicit_mdp_loss(states, actions_input, rewards, next_states, next_actions_input, dones)

        # --- Optimization ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient clipping might be useful here too
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        # --- Update Target Network ---
        self._update_target_network()

    def _compute_explicit_mdp_loss(self, states, actions, rewards, next_states, next_actions, dones) -> torch.Tensor:
        """Compute loss enforcing linear dynamics (M) and rewards (theta)."""

        # --- Compute features phi(s, a) ---
        phi_sa = self.phi_network(states, actions)

        # --- Reward Prediction Loss ---
        # Predict rewards: r_hat = theta^T * phi(s, a)
        predicted_rewards = self.reward_head(phi_sa)
        # MSE loss for rewards
        reward_loss = nn.functional.mse_loss(predicted_rewards, rewards)

        # --- Dynamics Prediction Loss ---
        # Predict next features using M: phi_hat' = M^T * phi(s, a)
        predicted_next_phi = self.M_matrix_layer(phi_sa)

        # Compute target features phi_target(s', a') (stop gradient)
        with torch.no_grad():
            phi_sa_next_target = self.phi_target_network(next_states, next_actions)
            # Target for dynamics loss: gamma * phi_target(s', a') * (1 - done)
            dynamics_target = self.ctrl_gamma * phi_sa_next_target * (1 - dones)

        # MSE loss for dynamics
        # Note: Contrastive loss (like InfoNCE from original CTRL) could be ADDED here
        #       if desired, operating directly on phi_sa and phi_sa_next_target.
        #       This implementation focuses only on the linear prediction losses.
        dynamics_loss = nn.functional.mse_loss(predicted_next_phi, dynamics_target)

        # --- Total Loss ---
        total_loss = (self.reward_loss_weight * reward_loss +
                      self.dynamics_loss_weight * dynamics_loss)

        # --- Logging (Optional) ---
        # print(f"Loss: {total_loss.item():.4f}, Rew Loss: {reward_loss.item():.4f}, Dyn Loss: {dynamics_loss.item():.4f}")

        return total_loss

    def _update_target_network(self):
        """Polyak averaging update for the phi_target_network."""
        # Only update phi_target_network, M and theta have no targets here
        with torch.no_grad():
            for param, target_param in zip(self.phi_network.parameters(), self.phi_target_network.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    # --- Methods to Access Parameters ---
    @torch.no_grad()
    def get_reward_params(self) -> np.ndarray:
        """
        Returns the learned reward parameter vector theta (shape: [feature_dim]).
        """
        # reward_head weights have shape [1, feature_dim] representing theta^T
        theta = self.reward_head.weight.data.squeeze().cpu().numpy()
        return theta

    @torch.no_grad()
    def get_transition_params(self) -> np.ndarray:
        """
        Returns the learned transition matrix M (shape: [feature_dim, feature_dim]).

        Note: M_matrix_layer.weight stores M^T with shape [feature_dim, feature_dim].
        We return the transpose to get M.
        """
        M_T = self.M_matrix_layer.weight.data.cpu().numpy()
        M = M_T.T
        return M

    def get_learned_components(self) -> Dict[str, nn.Module]:
         """ Returns the learned neural network components """
         return {
              "phi_network": self.phi_network,
              "M_matrix_layer": self.M_matrix_layer, # Layer representing M^T
              "reward_head": self.reward_head      # Layer representing theta^T
         }

    # --- save/load need to be updated to include M and theta heads ---
    def save(self, path: str):
        """Save the learned networks and optimizer state."""
        torch.save({
            'phi_network_state_dict': self.phi_network.state_dict(),
            'M_matrix_layer_state_dict': self.M_matrix_layer.state_dict(), # Added
            'reward_head_state_dict': self.reward_head.state_dict(),       # Added
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self._step_count,
        }, path)
        print(f"Wrapper state saved to {path}")

    def load(self, path: str):
        """Load the learned networks and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.phi_network.load_state_dict(checkpoint['phi_network_state_dict'])
        self.M_matrix_layer.load_state_dict(checkpoint['M_matrix_layer_state_dict']) # Added
        self.reward_head.load_state_dict(checkpoint['reward_head_state_dict'])       # Added
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._step_count = checkpoint.get('step_count', 0)

        self.phi_target_network.load_state_dict(self.phi_network.state_dict())
        self.phi_target_network.eval()
        print(f"Wrapper state loaded from {path}")


# --- Example Usage (using V2 wrapper) ---
if __name__ == "__main__":
    print("--- Testing V2 Wrapper with CartPole-v1 ---")
    base_env = gym.make("CartPole-v1")

    # Use the modified wrapper V2
    linear_mdp_env = LinearMDPLearnerWrapperV2(
        base_env,
        feature_dim=64,
        hidden_dim=128,
        learning_rate=5e-4, # Adjusted LR might be needed
        batch_size=128,
        buffer_size=50000,
        learning_starts=500,
        train_freq=4,
        gradient_steps=1,
        dynamics_loss_weight=1.0, # Can tune these weights
        reward_loss_weight=0.5,   # Example: down-weight reward loss
        device="auto"
    )

    # --- Interaction loop (same as before) ---
    num_episodes = 50
    total_steps = 0
    print_interval = 10

    for episode in range(num_episodes):
        obs, info = linear_mdp_env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        while not done:
            action = linear_mdp_env.action_space.sample()
            next_obs, reward, terminated, truncated, info = linear_mdp_env.step(action)
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1

        if (episode + 1) % print_interval == 0:
             print(f"Episode: {episode+1}/{num_episodes}, Steps: {episode_steps}, Total Steps: {total_steps}, Reward: {episode_reward:.2f}")

    print("\n--- Training Finished (V2 Wrapper) ---")

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
    print("  phi_network:", components['phi_network'])
    print("  M_matrix_layer (represents M^T):", components['M_matrix_layer'])
    print("  reward_head (represents theta^T):", components['reward_head'])

    linear_mdp_env.close()