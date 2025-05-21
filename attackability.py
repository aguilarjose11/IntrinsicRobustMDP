#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:52:50 2025

@author: joseaguilar
"""



import numpy as np
import cvxpy as cp
import torch

# TODO: Ensure needed funcs included, and deal with d(s) todo
def calculate_M_pi_dagger(phi_features, 
                          pi_target, 
                          relevant_states, 
                          gamma, 
                          action_space, 
                          mu_approx, 
                          transition_samples, 
                          d, 
                          reg=1e-6):
    """
    Helper function to estimate Sigma_phiphi, Sigma_phiphi_prime and compute M_pi_dagger.
    This is a simplified example assuming sufficient samples are available.
    
    Args:
        phi_features (dict)       : {(s, a): np.array} feature vectors.
        pi_target (dict)          : {s: a} target policy.
        relevant_states (list/set): States s with d^{\pi^\dagger}(s) > 0.
        gamma (float)             : Discount factor.
        action_space (list/set)   : All possible actions.
        mu_approx (dict)          : {s: probability} approximation of d^{\pi^\dagger}(s). Should sum to 1 over relevant_states.
                                    Alternatively, could pass samples drawn from d^{\pi^\dagger}.
        transition_samples (dict) : {(s, a): list_of_next_states_s_prime} or a function P(s,a).
                                    Needed to estimate expectation over s'.
        d (int)                   : Feature dimension.
        reg (float)               : Regularization parameter for matrix inversion.

    Returns:
        np.ndarray: The computed M_pi_dagger matrix (d, d).
        
    Note: This estimation requires careful implementation based on how mu_approx and transitions are provided.
          The example below assumes mu_approx provides weights and transition_samples provides next states.
          A more robust implementation would handle sampling directly if needed.
    """
    Sigma_phiphi = np.zeros((d, d))
    Sigma_phiphi_prime = np.zeros((d, d))
    
    # total_prob = sum(mu_approx.get(s, 0) for s in relevant_states) # Normalize if needed
    # if total_prob == 0:
    #     raise ValueError("Occupancy measure mu_approx sums to zero over relevant states.")

    for s in relevant_states:
        # prob_s = mu_approx.get(s, 0) / total_prob
        # if prob_s == 0:
        #     continue
            
        a = pi_target.get(s)
        if a is None:
            # Handle cases where policy is not defined for a relevant state (should ideally not happen)
            continue 
            
        phi_sa = phi_features(s, a)
        if phi_sa is None:
            # Handle missing features (should ideally not happen)
            continue
            
        # Estimate Sigma_phiphi contribution for state s (action a is fixed by pi_target)
        # Weight by d(s) = mu_approx(s). Assuming d(s,a) = d(s) * I(a=pi_target(s))
        Sigma_phiphi += np.outer(phi_sa, phi_sa) # prob_s * np.outer(phi_sa, phi_sa)

        # Estimate Sigma_phiphi_prime contribution
        # Requires sampling/averaging over s' ~ P(s,a) and a' = pi_target(s')
        next_states = transition_samples(s, a) # TODO: Convert to use mu
        if not next_states:
             # If no transition samples, cannot compute Sigma_phiphi_prime accurately. 
             # Could make assumptions (e.g., deterministic transitions) or raise error.
             # Here, we skip the contribution if no next states are known.
             print(f"Warning: No transition samples found for state {s}, action {a}. Sigma_phiphi_prime may be inaccurate.")
             continue

        avg_phi_s_prime_a_prime = np.zeros(d)
        num_samples = len(next_states)
        for s_prime in next_states:
            a_prime = pi_target.get(s_prime)
            if a_prime is None: 
                # If policy undefined for s_prime, assume zero features or handle differently
                continue
            phi_s_prime_a_prime = phi_features.get((s_prime, a_prime))
            if phi_s_prime_a_prime is None:
                # Handle missing features for next state-action
                continue
            avg_phi_s_prime_a_prime += phi_s_prime_a_prime
            
        if num_samples > 0:
             avg_phi_s_prime_a_prime /= num_samples
             Sigma_phiphi_prime += np.outer(phi_sa, avg_phi_s_prime_a_prime) # prob_s * np.outer(phi_sa, avg_phi_s_prime_a_prime)

    # Compute M_pi_dagger = (Sigma_phiphi - gamma * Sigma_phiphi_prime)^(-1) * Sigma_phiphi
    try:
        matrix_to_invert = Sigma_phiphi - gamma * Sigma_phiphi_prime
        # Add regularization for stability
        matrix_to_invert += reg * np.identity(d) 
        inv_matrix = np.linalg.inv(matrix_to_invert)
        M_pi_dagger = inv_matrix @ Sigma_phiphi
    except np.linalg.LinAlgError:
        print("Warning: Matrix inversion failed. Using pseudo-inverse.")
        matrix_to_invert = Sigma_phiphi - gamma * Sigma_phiphi_prime
        pinv_matrix = np.linalg.pinv(matrix_to_invert) # Use pseudo-inverse as fallback
        M_pi_dagger = pinv_matrix @ Sigma_phiphi
        
    return M_pi_dagger

# Original. Do not modify

def _one_hot_encode_action(action: int, action_dim: int) -> np.ndarray:
    """Convert discrete action to one-hot vector."""
    one_hot = np.zeros(action_dim, dtype=np.float32)
    one_hot[action] = 1.0
    return one_hot

@torch.no_grad()
def characterize_robustness_torch(phi_features, 
                                  pi_target, 
                                  theta_star, 
                                  relevant_states, 
                                  action_space, 
                                  M_pi_dagger,
                                  d,
                                  action_dim,
                                  solver=None,
                                  device='cpu'):
    """
    Solves the optimization program to characterize robustness for a linear MDP.

    Args:
        phi_features (dict): {(s, a): np.array} feature vectors.
        pi_target (dict): {s: a} target policy.
        theta_star (np.ndarray): Original reward parameter vector (d,).
        relevant_states (list/set): States s where d^{\pi^\dagger}(s) > 0.
        action_space (list/set): All possible actions a.
        M_pi_dagger (np.ndarray): Pre-computed matrix M = (Sigma_phiphi - gamma * Sigma_phiphi')^{-1} * Sigma_phiphi (d, d).
        solver (str, optional): Specify a CVXPY solver (e.g., 'ECOS', 'SCS', 'MOSEK'). Defaults to None (CVXPY default).

    Returns:
        tuple: (optimal_epsilon, optimal_theta_dagger, problem_status)
               optimal_epsilon (float): The maximum Q-function gap epsilon*. None if problem not solved optimally.
               optimal_theta_dagger (np.ndarray): The corresponding adversarial parameter theta_dagger. None if problem not solved optimally.
               problem_status (str): The status reported by the CVXPY solver.
    """
    # d = theta_star.shape[-1] # Or pass directly?
    if M_pi_dagger.shape!= (d, d):
        raise ValueError(f"Shape mismatch: M_pi_dagger shape {M_pi_dagger.shape} does not match feature dimension {d}")

    # Define optimization variables
    theta_dagger = cp.Variable(d, name="theta_dagger")
    epsilon = cp.Variable(name="epsilon")

    # Define constraints
    constraints = []

    # Constraint 1: Optimality of pi_target under theta_dagger
    for s in relevant_states:
        # Or use 
        # target_action = pi_target(linear_mdp_env._preprocess_obs(s)).detach().mean.argmax().cpu().numpy()
        target_action, _ = pi_target(s, return_vals=True)
        if target_action is None:
            print(f"Warning: Target policy pi_target not defined for relevant state {s}. Skipping constraints for this state.")
            continue
            
        # Or use: Note: Potentially use one-hot encoding
        # phi_target = phi_features(s, target_action)
        # if action_dim != 
        target_action_vec = target_action
        phi_target = phi_features(s.to(device), target_action_vec.to(device))
        if phi_target is None:
             print(f"Warning: Features phi(s, pi_target(s)) not found for state {s}, action {target_action}. Skipping constraints.")
             continue

        # TODO: Ensure this is valid with Gymnasium's spaces
        # IDEA: Use random sampling (uniform) for continuous spaces
        for a in action_space:
            if a == target_action.detach().cpu().argmax():
                continue

            # Or use: Note: Potentially use one-hot encoding
            # phi_alt = phi_features(s, a)
            a_vec = torch.Tensor(_one_hot_encode_action(a, action_dim))
            phi_alt = phi_features(s.to(device), a_vec.to(device))
            if phi_alt is None:
                print(f"Warning: Features phi(s, a) not found for state {s}, action {a}. Skipping constraint.")
                continue

            phi_diff = (phi_target - phi_alt).cpu().numpy()
            # v_sa = phi_diff^T @ M_pi_dagger
            v_sa = phi_diff @ M_pi_dagger # In numpy, @ handles row_vec @ matrix correctly
            
            constraints.append(v_sa @ theta_dagger - epsilon >= 0)

    # Constraint 2: Unchanged rewards for pi_target actions
    for s in relevant_states:
        # Or use 
        # target_action = pi_target(linear_mdp_env._preprocess_obs(s)).detach().mean.argmax().cpu().numpy()
        target_action, _ = pi_target(s, return_vals=True)
        if target_action is None:
            # Already warned above
            continue
            
        # Or use: Note: Potentially use one-hot encoding
        # phi_target = phi_features(s, target_action)
        target_action_vec = target_action
        # target_action_vec = torch.Tensor(_one_hot_encode_action(target_action, action_dim))
        
        u_s = phi_features(s.to(device), target_action_vec.to(device))
        if u_s is None:
            # Already warned above
            continue

        u_s = u_s.cpu().numpy()
        c_s = u_s @ theta_star
        constraints.append(u_s @ theta_dagger == c_s)

    # Constraint 3: theta^dagger L2 norm
    # Note: This upper-bound helps optimizer converge!
    constraints.append(cp.norm(theta_dagger, 2) <= np.sqrt(d))
    # Define objective
    objective = cp.Maximize(epsilon)

    # Define and solve the problem
    problem = cp.Problem(objective, constraints)
    
    try:
        # For options, see https://www.cvxpy.org/tutorial/solvers/index.html
        optimal_epsilon = problem.solve(solver=solver, verbose=True, eps=5e-3)
        # We multiply by 1 because of constraint:
        # v_sa @ theta_dagger - epsilon >= 0
        optimal_epsilon *= -1
        status = problem.status
        
        if status not in ["infeasible", "unbounded"] or status in ["optimal"]:
            optimal_theta_dagger = theta_dagger.value
            # Ensure epsilon is a float, handle potential None from solver errors
            optimal_epsilon = float(optimal_epsilon) if optimal_epsilon is not None else None
        else:
            optimal_epsilon = None
            optimal_theta_dagger = None
            
        return optimal_epsilon, optimal_theta_dagger, status

    except cp.error.SolverError as e:
        print(f"CVXPY Solver Error: {e}")
        return None, None, "Solver Error"
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        return None, None, "Error"


# def characterize_robustness(phi_features, 
#                             pi_target, 
#                             theta_star, 
#                             relevant_states, 
#                             action_space, 
#                             M_pi_dagger, 
#                             solver=None):
#     """
#     Solves the optimization program to characterize robustness for a linear MDP.

#     Args:
#         phi_features (dict): {(s, a): np.array} feature vectors.
#         pi_target (dict): {s: a} target policy.
#         theta_star (np.ndarray): Original reward parameter vector (d,).
#         relevant_states (list/set): States s where d^{\pi^\dagger}(s) > 0.
#         action_space (list/set): All possible actions a.
#         M_pi_dagger (np.ndarray): Pre-computed matrix M = (Sigma_phiphi - gamma * Sigma_phiphi')^{-1} * Sigma_phiphi (d, d).
#         solver (str, optional): Specify a CVXPY solver (e.g., 'ECOS', 'SCS', 'MOSEK'). Defaults to None (CVXPY default).

#     Returns:
#         tuple: (optimal_epsilon, optimal_theta_dagger, problem_status)
#                optimal_epsilon (float): The maximum Q-function gap epsilon*. None if problem not solved optimally.
#                optimal_theta_dagger (np.ndarray): The corresponding adversarial parameter theta_dagger. None if problem not solved optimally.
#                problem_status (str): The status reported by the CVXPY solver.
#     """
#     d = theta_star.shape
#     if M_pi_dagger.shape!= (d, d):
#         raise ValueError(f"Shape mismatch: M_pi_dagger shape {M_pi_dagger.shape} does not match feature dimension {d}")

#     # Define optimization variables
#     theta_dagger = cp.Variable(d, name="theta_dagger")
#     epsilon = cp.Variable(name="epsilon")

#     # Define constraints
#     constraints = []

#     # Constraint 1: Optimality of pi_target under theta_dagger
#     for s in relevant_states:
#         # Or use 
#         # target_action = pi_target(linear_mdp_env._preprocess_obs(s)).detach().mean.argmax().cpu().numpy()
#         target_action = pi_target.get(s)
#         if target_action is None:
#             print(f"Warning: Target policy pi_target not defined for relevant state {s}. Skipping constraints for this state.")
#             continue
            
#         # Or use: Note: Potentially use one-hot encoding
#         # phi_target = phi_features(s, target_action)
#         phi_target = phi_features.get((s, target_action))
#         if phi_target is None:
#              print(f"Warning: Features phi(s, pi_target(s)) not found for state {s}, action {target_action}. Skipping constraints.")
#              continue

#         # TODO: Ensure this is valid with Gymnasium's spaces
#         for a in action_space:
#             if a == target_action:
#                 continue

#             # Or use: Note: Potentially use one-hot encoding
#             # phi_alt = phi_features(s, a)
#             phi_alt = phi_features.get((s, a))
#             if phi_alt is None:
#                 print(f"Warning: Features phi(s, a) not found for state {s}, action {a}. Skipping constraint.")
#                 continue

#             phi_diff = phi_target - phi_alt
#             # v_sa = phi_diff^T @ M_pi_dagger
#             v_sa = phi_diff @ M_pi_dagger # In numpy, @ handles row_vec @ matrix correctly
            
#             constraints.append(v_sa @ theta_dagger - epsilon >= 0)

#     # Constraint 2: Unchanged rewards for pi_target actions
#     for s in relevant_states:
#         # Or use 
#         # target_action = pi_target(linear_mdp_env._preprocess_obs(s)).detach().mean.argmax().cpu().numpy()
#         target_action = pi_target.get(s)
#         if target_action is None:
#             # Already warned above
#             continue
            
#         # Or use: Note: Potentially use one-hot encoding
#         # phi_target = phi_features(s, target_action)
#         u_s = phi_features.get((s, target_action))
#         if u_s is None:
#             # Already warned above
#             continue

#         c_s = u_s @ theta_star
#         constraints.append(u_s @ theta_dagger == c_s)

#     # Define objective
#     objective = cp.Maximize(epsilon)

#     # Define and solve the problem
#     problem = cp.Problem(objective, constraints)
    
#     try:
#         optimal_epsilon = problem.solve(solver=solver)
#         status = problem.status
        
#         if status not in ["infeasible", "unbounded"] or status in ["optimal"]:
#             optimal_theta_dagger = theta_dagger.value
#             # Ensure epsilon is a float, handle potential None from solver errors
#             optimal_epsilon = float(optimal_epsilon) if optimal_epsilon is not None else None
#         else:
#             optimal_epsilon = None
#             optimal_theta_dagger = None
            
#         return optimal_epsilon, optimal_theta_dagger, status

#     except cp.error.SolverError as e:
#         print(f"CVXPY Solver Error: {e}")
#         return None, None, "Solver Error"
#     except Exception as e:
#         print(f"An unexpected error occurred during optimization: {e}")
#         return None, None, "Error"
    
# TODO: Add no_grad decorator?
# def characterize_robustness2(lin_mdp, 
#                              pi_target, 
#                              relevant_states, 
#                              solver=None):
#     """
#     Solves the optimization program to characterize robustness for a linear MDP.

#     Args:
#         phi_features (dict): {(s, a): np.array} feature vectors.
#         pi_target (dict): {s: a} target policy.
#         theta_star (np.ndarray): Original reward parameter vector (d,).
#         relevant_states (list/set): States s where d^{\pi^\dagger}(s) > 0.
#         action_space (list/set): All possible actions a.
#         M_pi_dagger (np.ndarray): Pre-computed matrix M = (Sigma_phiphi - gamma * Sigma_phiphi')^{-1} * Sigma_phiphi (d, d).
#         d (int): Dimensions of parameters.
#         solver (str, optional): Specify a CVXPY solver (e.g., 'ECOS', 'SCS', 'MOSEK'). Defaults to None (CVXPY default).

#     Returns:
#         tuple: (optimal_epsilon, optimal_theta_dagger, problem_status)
#                optimal_epsilon (float): The maximum Q-function gap epsilon*. None if problem not solved optimally.
#                optimal_theta_dagger (np.ndarray): The corresponding adversarial parameter theta_dagger. None if problem not solved optimally.
#                problem_status (str): The status reported by the CVXPY solver.
#     """
#     # if M_pi_dagger.shape != (d, d):
#     #     raise ValueError(f"Shape mismatch: M_pi_dagger shape {M_pi_dagger.shape} does not match feature dimension {d}")
#     d = lin_mdp.feature_dim
#     components = lin_mdp.get_learned_components()
#     phi_features = components['phi']
    
#     # Define optimization variables
#     theta_dagger = cp.Variable(d, name="theta_dagger")
#     epsilon = cp.Variable(name="epsilon")

#     # Define constraints
#     constraints = []

#     # Constraint 1: Optimality of pi_target under theta_dagger
#     target_action = pi_target(lin_mdp._preprocess_obs(relevant_states)).detach().mean.argmax().cpu().numpy()
#     phi_target = phi_features(relevant_states, target_action)
#     rel_state_alt_action = torch.cartesian_prod(relevant_states, lin_mdp.action_space)
#     # Split cartesian prod along columns (state and possible action)
#     phi_alt = phi_features(rel_state_alt_action[:,0], rel_state_alt_action[:,1])
    
    
#     #####
#     for s in relevant_states:
#         # Or use 
#         # target_action = pi_target(linear_mdp_env._preprocess_obs(s)).detach().mean.argmax().cpu().numpy()
#         target_action = pi_target.get(s)
#         if target_action is None:
#             print(f"Warning: Target policy pi_target not defined for relevant state {s}. Skipping constraints for this state.")
#             continue
            
#         # Or use: Note: Potentially use one-hot encoding
#         # phi_target = phi_features(s, target_action)
#         phi_target = phi_features.get((s, target_action))
#         if phi_target is None:
#              print(f"Warning: Features phi(s, pi_target(s)) not found for state {s}, action {target_action}. Skipping constraints.")
#              continue

#         # TODO: Ensure this is valid with Gymnasium's spaces
#         for a in action_space:
#             if a == target_action:
#                 continue

#             # Or use: Note: Potentially use one-hot encoding
#             # phi_alt = phi_features(s, a)
#             phi_alt = phi_features.get((s, a))
#             if phi_alt is None:
#                 print(f"Warning: Features phi(s, a) not found for state {s}, action {a}. Skipping constraint.")
#                 continue

#             phi_diff = phi_target - phi_alt
#             # v_sa = phi_diff^T @ M_pi_dagger
#             v_sa = phi_diff @ M_pi_dagger # In numpy, @ handles row_vec @ matrix correctly
            
#             constraints.append(v_sa @ theta_dagger - epsilon >= 0)

#     # Constraint 2: Unchanged rewards for pi_target actions
#     for s in relevant_states:
#         # Or use 
#         # target_action = pi_target(linear_mdp_env._preprocess_obs(s)).detach().mean.argmax().cpu().numpy()
#         target_action = pi_target.get(s)
#         if target_action is None:
#             # Already warned above
#             continue
            
#         # Or use: Note: Potentially use one-hot encoding
#         # phi_target = phi_features(s, target_action)
#         u_s = phi_features.get((s, target_action))
#         if u_s is None:
#             # Already warned above
#             continue

#         c_s = u_s @ theta_star
#         constraints.append(u_s @ theta_dagger == c_s)

#     # Define objective
#     objective = cp.Maximize(epsilon)

#     # Define and solve the problem
#     problem = cp.Problem(objective, constraints)
    
#     try:
#         optimal_epsilon = problem.solve(solver=solver)
#         status = problem.status
        
#         if status not in ["infeasible", "unbounded"] or status in ["optimal"]:
#             optimal_theta_dagger = theta_dagger.value
#             # Ensure epsilon is a float, handle potential None from solver errors
#             optimal_epsilon = float(optimal_epsilon) if optimal_epsilon is not None else None
#         else:
#             optimal_epsilon = None
#             optimal_theta_dagger = None
            
#         return optimal_epsilon, optimal_theta_dagger, status

#     except cp.error.SolverError as e:
#         print(f"CVXPY Solver Error: {e}")
#         return None, None, "Solver Error"
#     except Exception as e:
#         print(f"An unexpected error occurred during optimization: {e}")
#         return None, None, "Error"

# # Un-block with CTRL+1
# from torch import Tensor
# import cvxpy as cp

# # f_{CP} in paper
# def characterize(pi_t, phi, mu, theta, w, buff) -> tuple[float, Tensor]:
#     """Characterize given linear MDP (see paper) using buffer.
    

#     Parameters
#     ----------
#     pi_t : TYPE
#         Adversarial policy $\pi^\dagger$.
#     phi : TYPE
#         \phi component of linear MDP. Vector of d length.
#     mu : TYPE
#         \mu component of linear MDP. Vector of d length.
#     theta : TYPE
#         Reward parameter: r(s,a) = <\phi(s,a), \theta>.
#     w : TYPE
#         Q-function parameter: Q(s,a) = <\phi(s,a), w>.
#     buff : tuple[set[Any], set[Any]]
#         Buffer of sets of states and actions.

#     Returns
#     -------
#     tuple[float, Tensor]
#         \epsilon and \theta^\dagger as in paper.

#     """
#     pass