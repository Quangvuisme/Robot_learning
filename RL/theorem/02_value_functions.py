"""
Simple RL Example 2: Value Functions & Bellman Equation
========================================================
Understanding: V(s), Q(s,a), and the Bellman recursive relationship

V^π(s) = Expected cumulative reward starting from state s, following policy π
Q^π(s,a) = Expected cumulative reward starting from state s, taking action a, then following π

Bellman Equation: The recursive relationship that lets us compute values
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict


# ============================================================================
# EXAMPLE: Simple 1D Grid World (Same as Example 1)
# ============================================================================

class SimpleGridWorld:
    """1D grid: 0-1-2-3-4, goal at 4"""
    
    def __init__(self, gamma: float = 0.9):
        """
        gamma (discount factor): How much we care about future rewards
        - gamma close to 1: Long-term thinking
        - gamma close to 0: Short-term thinking
        """
        self.gamma = gamma
        self.goal = 4
        
    def get_next_states(self, state: int) -> Dict[str, int]:
        """Get possible next states for each action"""
        next_states = {}
        
        if state - 1 >= 0:
            next_states["LEFT"] = state - 1
        
        next_states["STAY"] = state
        
        if state + 1 <= 4:
            next_states["RIGHT"] = state + 1
        
        return next_states
    
    def get_reward(self, state: int, action: str, next_state: int) -> float:
        """Reward for taking action from state to next_state"""
        if next_state == self.goal:
            return +10  # Bonus for reaching goal
        elif next_state == state:
            # Tried to go out of bounds
            if (action == "LEFT" and state == 0) or (action == "RIGHT" and state == 4):
                return -10  # Penalty
        
        return -1  # Step cost


# ============================================================================
# EXAMPLE 1: POLICY EVALUATION (Computing V^π for a given policy)
# ============================================================================

def policy_evaluation(env: SimpleGridWorld, 
                     policy: Dict[int, str],
                     theta: float = 1e-4,
                     max_iterations: int = 100) -> Dict[int, float]:
    """
    Compute value function V^π(s) for a given policy.
    
    This is how we calculate: "How good is a policy?"
    
    Algorithm:
        Initialize V arbitrarily
        Repeat until convergence:
            For each state s:
                V(s) = E[R(s,π(s)) + γ * V(s')]  (Bellman equation)
    
    Returns: Dictionary mapping state -> value
    """
    
    # Initialize values
    V = {s: 0.0 for s in range(5)}
    V_history = [dict(V)]  # Track history
    
    for iteration in range(max_iterations):
        delta = 0  # Track change
        
        for state in range(5):
            old_value = V[state]
            
            # Get action from policy
            action = policy[state]
            
            # Get possible next state
            next_states = env.get_next_states(state)
            
            # Check if action is possible
            if action not in next_states:
                V[state] = 0
                continue
            
            next_state = next_states[action]
            reward = env.get_reward(state, action, next_state)
            
            # BELLMAN EQUATION: V(s) = R + γ * V(s')
            V[state] = reward + env.gamma * V[next_state]
            
            # Track change
            delta = max(delta, abs(old_value - V[state]))
        
        V_history.append(dict(V))
        
        # Converged?
        if delta < theta:
            print(f"  Converged in {iteration+1} iterations")
            break
    
    return V, V_history


# ============================================================================
# EXAMPLE 2: ACTION VALUE FUNCTION Q(s,a)
# ============================================================================

def compute_q_values(env: SimpleGridWorld,
                     V: Dict[int, float]) -> Dict[tuple, float]:
    """
    Compute Q values for all state-action pairs.
    
    Q^π(s,a) = R(s,a,s') + γ * V(s')
    
    This tells us: "What is the value of taking action a in state s?"
    """
    Q = {}
    
    for state in range(5):
        next_states = env.get_next_states(state)
        
        for action, next_state in next_states.items():
            reward = env.get_reward(state, action, next_state)
            # Q-value = immediate reward + discounted future value
            q_val = reward + env.gamma * V[next_state]
            Q[(state, action)] = q_val
    
    return Q


# ============================================================================
# EXAMPLE 3: Different Policies
# ============================================================================

def greedy_right_policy() -> Dict[int, str]:
    """Always move right (if possible)"""
    return {0: "RIGHT", 1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "STAY"}


def stay_policy() -> Dict[int, str]:
    """Always stay"""
    return {0: "STAY", 1: "STAY", 2: "STAY", 3: "STAY", 4: "STAY"}


def balanced_policy() -> Dict[int, str]:
    """Right most of the time"""
    return {0: "RIGHT", 1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "STAY"}


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_values(V: Dict[int, float], title: str):
    """Show value function visually"""
    states = sorted(V.keys())
    values = [V[s] for s in states]
    
    print(f"\n  {title}")
    print("  " + "-" * 50)
    
    # Bar chart
    for state in states:
        val = V[state]
        bar_length = int(max(0, val * 3))  # Scale for display
        bar = "█" * bar_length if val >= 0 else "▓" * abs(int(val))
        print(f"    State {state}: {bar:20s} V={val:+7.2f}")


# ============================================================================
# Main: Run examples
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("VALUE FUNCTIONS & BELLMAN EQUATION")
    print("="*70)
    
    env = SimpleGridWorld(gamma=0.9)
    
    # ========== EXAMPLE 1: Greedy Right Policy ==========
    print("\n1. EVALUATE 'ALWAYS GO RIGHT' POLICY")
    print("-" * 70)
    
    policy = greedy_right_policy()
    print("Policy: ", policy)
    print("Computing V^π(s) using Bellman equation...")
    
    V_right, history = policy_evaluation(env, policy)
    visualize_values(V_right, "State Values V(s)")
    
    # Show Bellman computation for one state
    print("\n  How Bellman equation works for State 2:")
    print("    Action: RIGHT → Next State: 3")
    print(f"    V(2) = R(2→3) + γ * V(3)")
    print(f"    V(2) = -1 + 0.9 * {V_right[3]:.2f}")
    print(f"    V(2) = {V_right[2]:.2f}")
    
    # ========== EXAMPLE 2: Q-values ==========
    print("\n2. Q-VALUES: 'What is value of each action?'")
    print("-" * 70)
    
    Q_right = compute_q_values(env, V_right)
    
    print("For each state, show Q(s,a) for each action:")
    for state in range(5):
        next_states = env.get_next_states(state)
        print(f"\n  State {state}:")
        for action in next_states:
            q_val = Q_right[(state, action)]
            print(f"    Q({state}, {action:6s}) = {q_val:+7.2f}")
    
    # ========== EXAMPLE 3: Compare Policies ==========
    print("\n3. COMPARE DIFFERENT POLICIES")
    print("-" * 70)
    
    policies = [
        ("Go Right", greedy_right_policy()),
        ("Always Stay", stay_policy()),
    ]
    
    for policy_name, policy in policies:
        print(f"\nPolicy: {policy_name}")
        V, _ = policy_evaluation(env, policy)
        visualize_values(V, f"Values for {policy_name}")
    
    # ========== EXAMPLE 4: Effect of Discount Factor ==========
    print("\n4. EFFECT OF DISCOUNT FACTOR γ")
    print("-" * 70)
    print("γ controls how much we value future rewards")
    print("γ=1: Value infinite future rewards equally")
    print("γ=0: Only care about immediate reward")
    
    policy = greedy_right_policy()
    
    for gamma in [0.1, 0.5, 0.9, 0.99]:
        env_gamma = SimpleGridWorld(gamma=gamma)
        V_gamma, _ = policy_evaluation(env_gamma, policy)
        print(f"\n  γ = {gamma}")
        print(f"    V(0) = {V_gamma[0]:+7.2f}  (state farthest from goal)")
        print(f"    V(4) = {V_gamma[4]:+7.2f}  (goal state)")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("• V(s) = Expected cumulative reward from state s")
    print("• Q(s,a) = Expected value of taking action a in state s")
    print("• Bellman equation: V(s) = R + γ * V(s') [recursive relationship]")
    print("• γ (gamma): How much we discount future rewards")
    print("• Different policies → Different value functions")
    print("="*70)
