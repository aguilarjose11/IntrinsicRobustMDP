#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:35:55 2025

@author: joseaguilar
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Wrapper for discretizing continuous action spaces.
    
    This wrapper converts a continuous action space into a discrete one by creating
    a grid of possible actions and mapping discrete actions to their continuous counterparts.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        n_bins_per_dim: Union[int, list] = 5,
        action_min: Optional[np.ndarray] = None,
        action_max: Optional[np.ndarray] = None
    ):
        """
        Initialize the discrete action wrapper.
        
        Args:
            env: The environment to wrap
            n_bins_per_dim: Number of bins for each action dimension (can be int or list)
            action_min: Optional minimum values for each action dimension
            action_max: Optional maximum values for each action dimension
        """
        super().__init__(env)
        
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("DiscreteActionWrapper can only wrap environments with Box action spaces")
        
        # Get action dimensions
        self.action_dim = env.action_space.shape[0]
        
        # Handle n_bins_per_dim as int or list
        if isinstance(n_bins_per_dim, int):
            self.n_bins = [n_bins_per_dim] * self.action_dim
        else:
            if len(n_bins_per_dim) != self.action_dim:
                raise ValueError(f"n_bins_per_dim list length ({len(n_bins_per_dim)}) "
                                f"must match action dimensions ({self.action_dim})")
            self.n_bins = n_bins_per_dim
        
        # Get action bounds
        if action_min is None:
            self.action_min = env.action_space.low
        else:
            self.action_min = action_min
            
        if action_max is None:
            self.action_max = env.action_space.high
        else:
            self.action_max = action_max
        
        # Create the discrete action space
        self.total_actions = np.prod(self.n_bins)
        self.action_space = gym.spaces.Discrete(self.total_actions)
        
        # Precompute action mappings
        self._create_action_mapping()
        
    def _create_action_mapping(self):
        """Create mapping from discrete actions to continuous action values."""
        self.action_grid = []
        
        for dim in range(self.action_dim):
            # Create linearly spaced points for this dimension
            dim_values = np.linspace(
                self.action_min[dim],
                self.action_max[dim],
                self.n_bins[dim]
            )
            self.action_grid.append(dim_values)
        
        # Create the full mapping of indices to continuous actions
        self.action_map = {}
        
        # Generate all possible combinations of indices
        indices_list = [range(self.n_bins[dim]) for dim in range(self.action_dim)]
        flat_idx = 0
        
        for indices in self._cartesian_product(indices_list):
            # Convert indices to continuous action
            continuous_action = np.array([self.action_grid[dim][idx] for dim, idx in enumerate(indices)])
            self.action_map[flat_idx] = continuous_action
            flat_idx += 1
    
    def _cartesian_product(self, arrays):
        """Generate all combinations of indices for multi-dimensional grid."""
        curr_indices = [0] * len(arrays)
        stop = False
        
        while not stop:
            yield curr_indices.copy()
            
            # Update indices
            for i in range(len(arrays) - 1, -1, -1):
                curr_indices[i] += 1
                if curr_indices[i] < len(arrays[i]):
                    break
                curr_indices[i] = 0
                if i == 0:
                    stop = True
    
    def action(self, discrete_action):
        """Map discrete action to continuous action."""
        if discrete_action >= self.total_actions:
            raise ValueError(f"Discrete action {discrete_action} is out of bounds (max: {self.total_actions-1})")
        
        return self.action_map[discrete_action]
    
    def reverse_action(self, continuous_action):
        """
        Map continuous action to closest discrete action.
        Note: This is approximate and primarily used for visualization or debugging.
        """
        # Find closest discrete action by computing distance to each entry in action_map
        min_dist = float('inf')
        best_action = 0
        
        for action_idx, action_val in self.action_map.items():
            dist = np.sum((continuous_action - action_val) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_action = action_idx
                
        return best_action


class DiscreteObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper for discretizing continuous observation spaces.
    
    This wrapper converts a continuous observation space into a discrete one by binning
    each dimension of the observation space.
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        n_bins_per_dim: Union[int, list] = 10,
        obs_min: Optional[np.ndarray] = None,
        obs_max: Optional[np.ndarray] = None
    ):
        """
        Initialize the discrete observation wrapper.
        
        Args:
            env: The environment to wrap
            n_bins_per_dim: Number of bins for each observation dimension (can be int or list)
            obs_min: Optional minimum values for each observation dimension
            obs_max: Optional maximum values for each observation dimension
        """
        super().__init__(env)
        
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("DiscreteObservationWrapper can only wrap environments with Box observation spaces")
        
        # Get observation dimensions
        self.obs_dim = env.observation_space.shape[0]
        
        # Handle n_bins_per_dim as int or list
        if isinstance(n_bins_per_dim, int):
            self.n_bins = [n_bins_per_dim] * self.obs_dim
        else:
            if len(n_bins_per_dim) != self.obs_dim:
                raise ValueError(f"n_bins_per_dim list length ({len(n_bins_per_dim)}) "
                                f"must match observation dimensions ({self.obs_dim})")
            self.n_bins = n_bins_per_dim
        
        # Get observation bounds
        if obs_min is None:
            self.obs_min = env.observation_space.low
            # Handle unbounded spaces (inf values)
            self.obs_min = np.where(self.obs_min == -np.inf, -10.0, self.obs_min)
        else:
            self.obs_min = obs_min
            
        if obs_max is None:
            self.obs_max = env.observation_space.high
            # Handle unbounded spaces (inf values)
            self.obs_max = np.where(self.obs_max == np.inf, 10.0, self.obs_max)
        else:
            self.obs_max = obs_max
        
        # Create the new observation space
        self.observation_space = gym.spaces.MultiDiscrete(self.n_bins)
        
    def observation(self, observation):
        """Convert continuous observation to discrete indices."""
        # Scale observation to be within [0, n_bins-1] for each dimension
        scaled_obs = []
        
        for dim in range(self.obs_dim):
            # Clip observation to be within bounds
            clipped_val = np.clip(
                observation[dim], 
                self.obs_min[dim], 
                self.obs_max[dim]
            )
            
            # Convert to bin index
            bin_idx = int(np.floor(
                (clipped_val - self.obs_min[dim]) / 
                (self.obs_max[dim] - self.obs_min[dim]) * 
                self.n_bins[dim]
            ))
            
            # Ensure index is within valid range
            bin_idx = min(bin_idx, self.n_bins[dim] - 1)
            scaled_obs.append(bin_idx)
        
        return np.array(scaled_obs, dtype=np.int64)


# Helper function to apply both wrappers at once
def make_discrete_env(
    env_id, 
    action_bins=5, 
    obs_bins=10, 
    action_min=None, 
    action_max=None,
    obs_min=None, 
    obs_max=None,
    seed=None
):
    """
    Create a discretized version of a continuous Gym environment.
    
    Args:
        env_id: The ID of the environment to create
        action_bins: Number of bins for each action dimension
        obs_bins: Number of bins for each observation dimension
        action_min: Optional minimum values for actions
        action_max: Optional maximum values for actions
        obs_min: Optional minimum values for observations
        obs_max: Optional maximum values for observations
        seed: Random seed
        
    Returns:
        A wrapped environment with discrete action and observation spaces
    """
    env = gym.make(env_id, render_mode=None)
    
    if seed is not None:
        env.reset(seed=seed)
    
    # Apply wrappers
    env = DiscreteObservationWrapper(
        env, 
        n_bins_per_dim=obs_bins,
        obs_min=obs_min,
        obs_max=obs_max
    )
    
    env = DiscreteActionWrapper(
        env, 
        n_bins_per_dim=action_bins,
        action_min=action_min,
        action_max=action_max
    )
    
    return env


# Usage examples
if __name__ == "__main__":
    # Example 1: Pendulum-v1
    pendulum_env = make_discrete_env(
        "Pendulum-v1", 
        action_bins=7,  # 7 discrete actions
        obs_bins=10,    # 10 bins per observation dimension
        seed=42
    )
    
    print(f"Pendulum env details:")
    print(f"  Original action space: Box(-2.0, 2.0, (1,), float32)")
    print(f"  Discretized action space: {pendulum_env.action_space}")
    print(f"  Original observation space: Box(-3.14159, 3.14159, (3,), float32)")
    print(f"  Discretized observation space: {pendulum_env.observation_space}")
    
    # Example 2: HalfCheetah-v4
    cheetah_env = make_discrete_env(
        "HalfCheetah-v4", 
        action_bins=[5, 5, 5, 5, 5, 5],  # 5 bins for each of the 6 action dimensions
        obs_bins=10,                      # 10 bins for each observation dimension
        seed=42
    )
    
    print(f"\nHalfCheetah env details:")
    print(f"  Original action space: Box(-1.0, 1.0, (6,), float32)")
    print(f"  Discretized action space: {cheetah_env.action_space}")
    print(f"  Original observation space: Box(-inf, inf, (17,), float32)")
    print(f"  Discretized observation space: {cheetah_env.observation_space}")
    
    # Example of running a simple episode
    obs, _ = cheetah_env.reset(seed=42)
    total_reward = 0
    
    for _ in range(100):
        # Take random discrete action
        action = cheetah_env.action_space.sample()
        obs, reward, terminated, truncated, info = cheetah_env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
            
    print(f"\nRandom Half-Cheetah episode reward: {total_reward}")