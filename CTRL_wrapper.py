#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 12:13:15 2025

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
import copy

from CTRL_networks import Critic, Phi, Mu, DiagGaussianActor
from util import to_np


class ReplayBuffer:
    """A simple FIFO experience replay buffer."""
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        action_processed = action
        if isinstance(action, int):
             action_processed = np.array([action])
        elif isinstance(action, (np.ndarray, list, tuple)):
             action_processed = np.array(action)

        experience = (state, action_processed, reward, next_state, float(done))
        self.buffer.append(experience)

    def sample(self, batch_size: int, device: torch.device):
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


class CTRLMDPWrapper(Wrapper):
    def __init__(self, 
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
                 ctrl_gamma: float = 0.99,          # Discount factor used in dynamics target
                 device: str = "auto",):
        """
        

        Parameters
        ----------
        env : gym.Env
            DESCRIPTION.
        feature_dim : int, optional
            DESCRIPTION. The default is 128.
        hidden_dim : int, optional
            DESCRIPTION. The default is 256.
        learning_rate : float, optional
            DESCRIPTION. The default is 3e-4.
        batch_size : int, optional
            DESCRIPTION. The default is 256.
        buffer_size : int, optional
            DESCRIPTION. The default is 1_000_000.
        train_freq : int, optional
            DESCRIPTION. The default is 1.
        gradient_steps : int, optional
            DESCRIPTION. The default is 1.
        learning_starts : int, optional
            DESCRIPTION. The default is 1000.
        tau : float, optional
            DESCRIPTION. The default is 0.005.
        dynamics_loss_weight : float, optional
            DESCRIPTION. The default is 1.0.
        reward_loss_weight : float, optional
            DESCRIPTION. The default is 1.0.
        ctrl_gamma : float, optional
            DESCRIPTION. The default is 0.99.
        device : str, optional
            DESCRIPTION. The default is "auto".

        Returns
        -------
        None.

        """
        super().__init__(env)
        
        ''' Hyperparameters '''
        self.original_en          = env
        self.feature_dim = self.d = feature_dim
        self.batch_size           = batch_size
        self.buffer_size          = buffer_size
        self.train_freq           = train_freq
        self.gradient_steps       = gradient_steps
        self.learning_starts      = learning_starts
        self.tau                  = tau
        self.dynamics_loss_weight = dynamics_loss_weight
        self.reward_loss_weight   = reward_loss_weight
        self.ctrl_gamma           = ctrl_gamma
        # self.alpha                = 1.
        # --- Device Setup ---
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        ''' State and Action Space Handling '''
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

        self.target_entropy = -self.action_dim
        # Make it between 0.5 and 1.0
        self.log_alpha = torch.tensor(np.log(alpha:=0.6)).to(self.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=learning_rate, betas=[0.9, 0.999])        

        ''' Model Networks '''
        # Feature extractor network
        # self.phi_network = PhiNetwork(self.state_dim, self.action_dim, feature_dim, hidden_dim).to(self.device)
        # self.phi_target_network = PhiNetwork(self.state_dim, self.action_dim, feature_dim, hidden_dim).to(self.device)
        # self.phi_target_network.load_state_dict(self.phi_network.state_dict())
        # self.phi_target_network.eval()
        
        self.phi = Phi(state_dim=self.state_dim, 
				 action_dim=self.action_dim, 
				 feature_dim=feature_dim, 
				 hidden_dim=hidden_dim).to(self.device)
        self.phi_target = copy.deepcopy(self.phi)
        self.phi_target.eval()

        # Explicit Transition Matrix (M^T represented by Linear layer weights)
        # Maps phi(s,a) -> predicted next phi(s',a') (before discount)
        self.M_matrix_layer = nn.Linear(feature_dim, feature_dim, bias=False).to(self.device)
        
        self.mu = Mu(state_dim=self.state_dim,                     
                     feature_dim=feature_dim, 
                     hidden_dim=hidden_dim).to(self.device)
        
        # self.theta = Theta(feature_dim=feature_dim).to(self.device)
        

        # Explicit Reward Weights (theta^T represented by Linear layer weights)
        # Maps phi(s,a) -> predicted reward
        self.reward_head = nn.Linear(feature_dim, 1, bias=False).to(self.device) # Often bias is excluded
        # Same as theta, but simpler!
        
        # Note that M and mu are highly related! See notes...
        self.optimizer = optim.AdamW(
            list(self.phi.parameters()) +
            list(self.M_matrix_layer.parameters()) +
            list(self.mu.parameters()) +
            list(self.reward_head.parameters()),
            lr=learning_rate
        )
        # --- Data Collection & Training State (same as before) ---
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self._step_count = 0
        self._last_obs = None
        self._current_trajectory_actions = []
        
        # frozen phi for critic/actor update
        self.frozen_phi = Phi(state_dim=self.state_dim, 
                              action_dim=self.action_dim, 
                              feature_dim=feature_dim, 
                              hidden_dim=hidden_dim).to(self.device)
        self.frozen_phi_target = copy.deepcopy(self.frozen_phi)
        
        # Actor Policy and Critic (Q-function)
        self.actor = DiagGaussianActor(
			obs_dim=self.state_dim, 
			action_dim=self.action_dim,
			hidden_dim=512,
			hidden_depth=1,
			log_std_bounds=[-5., 2.],).to(self.device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), weight_decay=0, lr=learning_rate, betas=[0.9, 0.999]) 	# lower lr for actor/alpha
        
        # Alpha temperature not needed.
        # self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate/3, betas=[0.9, 0.999])

        self.critic = Critic(feature_dim=feature_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), weight_decay=0, lr=learning_rate, betas=[0.9, 0.999])
        
        
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
     
    ####################
    ### Core Methods ###
    ####################
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    @torch.no_grad()
    def get_feature(self, obs: np.ndarray, action):
        """Compute the learned feature vector phi(s, a)."""
        self.phi.eval()
        state_tensor = self._preprocess_obs(obs)
        processed_action = self._action_processor(action)
        action_tensor = torch.tensor(processed_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_input = self._process_action_for_input(action_tensor)
        feature_tensor = self.phi(state_tensor, action_input)
        self.phi.train()
        return feature_tensor.squeeze(0).cpu().numpy()
    
    
    def step(self, action, learning=True):
        """Take step, store transition, potentially train."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        done = terminated or truncated

        if self._last_obs is not None:
            processed_action_storage = self._action_processor(action)
            self.replay_buffer.add(self._last_obs, processed_action_storage, reward, obs, done)
            self._current_trajectory_actions.append(processed_action_storage)

        if self._step_count >= self.learning_starts and self._step_count % self.train_freq == 0:
            if len(self.replay_buffer) >= self.batch_size and learning:
                for _ in range(self.gradient_steps):
                    self.train_representation() # Training step.

        self._last_obs = obs
        if done:
           # info['final_observation'] = self._last_obs # Gymnasium standard
           if hasattr(self, '_last_obs') and self._last_obs is not None:
               info['final_observation'] = np.array(self._last_obs, dtype=self.observation_space.dtype) # Ensure correct dtype for Gym standard
           self._current_trajectory_actions = []

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reset environment and internal state."""
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._current_trajectory_actions = []
        return obs, info
    
    ################
    ### Training ###
    ################
    
    def train_representation(self):
        self.phi.train()
        self.M_matrix_layer.train()
        self.mu.train()
        self.reward_head.train()
        self.critic.train()
        self.actor.train()
        # train features: phi, mu/M, theta
        for _ in range(3+1):
            batch = self.replay_buffer.sample(self.batch_size, self.device)
            # Step features
            # Train phi, M, reward_head (theta), mu (TODO!)
            self.feature_step(batch)
            # Step target features
            self.update_feature_target()
       
        # update frozen phi with new phi
        self.frozen_phi.load_state_dict(self.phi.state_dict().copy())
        self.frozen_phi_target.load_state_dict(self.phi.state_dict().copy())
        
        # Train critic (Q-function)
        self.critic_step(batch)
        # Train actor: update_actor_and_alpha in other implementation.
        self.actor_step(batch)
        # Update frozen models (actor and maybe critic.)
        self.update_target()
        # NOTE: If using step info, return it here as dict!
        self.phi.eval()
        self.M_matrix_layer.eval()
        self.mu.eval()
        self.reward_head.eval()
        self.critic.eval()
        self.actor.eval()
        return None
       
    def feature_step(self, batch):
        state, action, reward, next_state, done = batch
        # Ensure state follow format
        # state = self._preprocess_obs(state)
        # next_state = self._preprocess_obs(next_state)
        # Ensure actions follow format
        # proc_action = self._action_processor(action.cpu())
        # action_tensor = torch.tensor(proc_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        # action = self._process_action_for_input(action_tensor)
        
        # Compute phi and mu
        z_phi = self.phi(state, action)
        z_mu_next = self.mu(next_state)
        
        assert z_phi.shape[-1] == self.feature_dim
        assert z_mu_next.shape[-1] == self.feature_dim
        
        labels = torch.eye(state.shape[0]).to(self.device)
        # we take NCE gamma = 1 here, the paper uses 0.2
        contrastive = (z_phi[:, None, :] * z_mu_next[None, :, :]).sum(-1) 
        model_loss = nn.CrossEntropyLoss()
        # Model loss -- w.r.t. phi, mu
        model_loss = model_loss(contrastive, labels)
        
        # Reward loss -- w.r.t. theta: reward_head
        r_loss = nn.functional.mse_loss(self.reward_head(z_phi), reward).mean()
    
        predicted_next_phi = self.M_matrix_layer(z_phi)
        # Compute target features phi_target(s', a') (stop gradient)
        with torch.no_grad():
            # next_action from mean of policy dist.
            # Note that due to no_grad, self.actor won't update
            dist = self.actor(state)
            next_action = dist.mean
            # next_action = self.actor.select_action(state, explore=False)
            phi_sa_next_target = self.phi_target(next_state, next_action)
            # Target for dynamics loss: gamma * phi_target(s', a') * (1 - done)
            dynamics_target = self.ctrl_gamma * phi_sa_next_target * (1 - done)
            
        # Dynamics loss -- w.r.t. matrix M ~ mu
        dynamics_loss = nn.functional.mse_loss(predicted_next_phi, dynamics_target).mean()
        ''' Least Squares Temporal Difference Q-learning
        # Remember to check no_grad is needed or not here.
                if stage_2_flag: # Stage 2 poisoning

        lbda = 0.1
        # (d x batch) @ (batch x d) --> (d x d)                if stage_2_flag: # Stage 2 poisoning

        phi_phi = z_phi @ z_phi.T
        # (d x batch) @ (batch x d)' --> (d x d)
        gmma_phi_phi_nxt = self.ctrl_gamma * (z_phi @ phi_sa_next_target.T)
        # (d x d) @ (d x 1) --> (d x 1)
        phi_r = self.reward_head(phi_phi)
        # [(d x d) - (d x d) + (d x d)] --> (d x d) ~!~ inverse
        # phi_td_rdg = torch.linalg.inv(phi_phi - gmma_phi_phi_nxt + lbda * torch.eye(self.feature_dim))
        phi_td = torch.linalg.pinv(phi_phi - gmma_phi_phi_nxt)
        
        # (d x d) x (d x 1) --> (d x 1)
        w = phi_td @ phi_r
        # Updating M_matrix_layer may require a different approach...

        '''

        # Optionally use weights on each loss.
        loss = model_loss\
             + self.reward_loss_weight * r_loss\
             + self.dynamics_loss_weight * dynamics_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_feature_target(self):
        """Polyak averaging update for the phi_target_network."""
        # Only update phi_target_network, M and theta have no targets here
        for param, target_param in zip(self.phi.parameters(), self.phi_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def critic_step(self, batch):
        state, action, reward, next_state, done = batch
        # Ensure state follow format
        # state = self._preprocess_obs(state)
        # next_state = self._preprocess_obs(next_state)
        # Ensure actions follow format
        # proc_action = self._action_processor(action.cpu())
        # action_tensor = torch.tensor(proc_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        # action = self._process_action_for_input(action_tensor)
        
        with torch.no_grad():
            dist = self.actor(next_state)
            # Note that actor outputs one-hot style
            # To get action, use argmax?
            next_action = dist.rsample()
            next_action_log_pi = dist.log_prob(next_action).sum(-1, keepdim=True)
            
            # Ensure actions follow format
            # proc_next_action= self._action_processor(next_action.cpu())
            # next_action_tensor = torch.tensor(proc_next_action, dtype=torch.float32, device=self.device).unsqueeze(0)
            # next_action = self._process_action_for_input(next_action_tensor)
            
            # Use frozen targets (which should have been updated already)
            z_phi = self.frozen_phi(state, action)
            z_phi_next = self.frozen_phi(next_state, next_action)
            
            # DoubleQ trick
            next_q1, next_q2 = self.critic_target(z_phi_next)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_log_pi
            target_q = reward + (1. - done) * self.ctrl_gamma * next_q 
        q1, q2 = self.critic(z_phi)
        q1_loss = nn.functional.mse_loss(target_q, q1)
        q2_loss = nn.functional.mse_loss(target_q, q2)
        q_loss = q1_loss + q2_loss
        # Apply update
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
    
    def actor_step(self, batch):
        state, action_, reward, next_state, done = batch
        # Ensure state follow format
        # state = self._preprocess_obs(state)
        # next_state = self._preprocess_obs(next_state)
        
        dist = self.actor(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # Ensure actions follow format
        # proc_action = self._action_processor(action)
        # action_tensor = torch.tensor(proc_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        # action = self._process_action_for_input(action_tensor)
        
        z_phi = self.frozen_phi(state, action)
        q1, q2 = self.critic(z_phi)
        q = torch.min(q1, q2)
        actor_loss = ((self.alpha) * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Updates alpha value.
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = ((self.alpha) * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        
        # See CTRL github code for learnable_temperature alpha
    
    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
    
    def get_learned_components(self):
         """ Returns the learned neural network components """
         return {
              "phi": self.phi,
              "M_matrix_layer": self.M_matrix_layer, # Layer representing M^T
              "mu": self.mu,
              "reward_head": self.reward_head      # Layer representing theta^T
         }
    
    # --- save/load need to be updated to include M and theta heads ---
    def save(self, path: str):
        """Save the learned networks and optimizer state."""
        torch.save({
            'phi_state_dict': self.phi.state_dict(),
            'M_matrix_layer_state_dict': self.M_matrix_layer.state_dict(), # Added
            'mu_state_dict': self.mu.state_dict(),
            'reward_head_state_dict': self.reward_head.state_dict(),       # Added
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'step_count': self._step_count,
        }, path)
        print(f"Wrapper state saved to {path}")
    
    def load(self, path: str):
        """Load the learned networks and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        # Load linear MDP paramterized functions
        self.phi.load_state_dict(checkpoint['phi_state_dict'])
        self.M_matrix_layer.load_state_dict(checkpoint['M_matrix_layer_state_dict']) # Added
        self.mu.load_state_dict(checkpoint['mu_state_dict'])
        self.reward_head.load_state_dict(checkpoint['reward_head_state_dict'])       # Added
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        # Load optimizers
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        # Load target networks. Frozen ones skiped since __init__ is enough.
        self.phi_target.load_state_dict(self.phi.state_dict())
        self.phi_target.eval()
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self._step_count = checkpoint.get('step_count', 0)
        print(f"Wrapper state loaded from {path}")
    
    





