"""
Simple RL Example 3: Policy Improvement & Optimal Policy
=========================================================
Finding the BEST policy using Q-values

Key idea: 
- If we know Q(s,a) for all state-action pairs
- We can find the best action by choosing: π*(s) = argmax_a Q(s,a)
- This is called greedy policy improvement
"""

import numpy as np
from typing import Dict, Tuple


# ============================================================================
# ENVIRONMENT
# ============================================================================

class GridWorld:
    """1D grid world: 0-1-2-3-4, goal at 4"""
    
    def __init__(self, gamma: float = 0.9):
        self.gamma = gamma
        self.goal = 4
        self.states = list(range(5))
        
    def get_transitions(self, state: int, action: str) -> Tuple[int, float]:
        """
        Get (next_state, reward) for taking action from state
        Deterministic environment
        """
        next_state = state
        
        if action == "LEFT" and state > 0:
            next_state = state - 1
        elif action == "RIGHT" and state < 4:
            next_state = state + 1
        elif action == "STAY":
            next_state = state
        
        # Reward
        if state == 4:  # Already at goal
            reward = 0
        elif next_state == self.goal:
            reward = +10
        elif next_state == state and action in ["LEFT", "RIGHT"]:  # Tried to go out of bounds
            reward = -10
        else:
            reward = -1
        
        return next_state, reward
    
    def get_possible_actions(self, state: int) -> list:
        """Get valid actions from state"""
        actions = ["STAY"]
        if state > 0:
            actions.append("LEFT")
        if state < 4:
            actions.append("RIGHT")
        return actions


# ============================================================================
# POLICY EVALUATION (Same as Example 2)
# ============================================================================

def policy_evaluation(env: GridWorld, 
                     policy: Dict[int, str],
                     max_iterations: int = 100,
                     theta: float = 1e-4) -> Dict[int, float]:
    """Compute V^π(s) for a given policy"""
    
    V = {s: 0.0 for s in env.states}
    
    for iteration in range(max_iterations):
        delta = 0
        
        for state in env.states:
            old_value = V[state]
            action = policy[state]
            next_state, reward = env.get_transitions(state, action)
            
            # Bellman: V(s) = R(s,a,s') + γ * V(s')
            V[state] = reward + env.gamma * V[next_state]
            
            delta = max(delta, abs(old_value - V[state]))
        
        if delta < theta:
            break
    
    return V


# ============================================================================
# COMPUTE Q-VALUES
# ============================================================================

def compute_q_values(env: GridWorld, V: Dict[int, float]) -> Dict[Tuple, float]:
    """Compute Q(s,a) for all state-action pairs"""
    
    Q = {}
    
    for state in env.states:
        for action in env.get_possible_actions(state):
            next_state, reward = env.get_transitions(state, action)
            # Q(s,a) = R(s,a,s') + γ * V(s')
            Q[(state, action)] = reward + env.gamma * V[next_state]
    
    return Q


# ============================================================================
# POLICY IMPROVEMENT
# ============================================================================

def policy_improvement(env: GridWorld, Q: Dict[Tuple, float]) -> Dict[int, str]:
    """
    Create a new policy by choosing best action in each state
    
    Greedy policy: π(s) = argmax_a Q(s,a)
    
    This means: "For each state, pick the action with highest Q-value"
    """
    
    new_policy = {}
    
    for state in env.states:
        # Get all possible actions
        possible_actions = env.get_possible_actions(state)
        
        # Find action with highest Q-value
        best_action = None
        best_q_value = -np.inf
        
        for action in possible_actions:
            q_value = Q[(state, action)]
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        new_policy[state] = best_action
    
    return new_policy


# ============================================================================
# POLICY ITERATION (Repeat: Evaluate → Improve)
# ============================================================================

def policy_iteration(env: GridWorld, 
                    initial_policy: Dict[int, str] = None,
                    max_iterations: int = 100,
                    verbose: bool = True) -> Dict[int, str]:
    """
    Find optimal policy using policy iteration
    
    Algorithm:
        1. Start with random policy
        2. Evaluate: Compute V^π(s) for current policy
        3. Improve: Generate new greedy policy using Q-values
        4. Repeat until policy converges
    """
    
    # Initialize policy
    if initial_policy is None:
        initial_policy = {s: "STAY" for s in env.states}
    
    policy = dict(initial_policy)
    
    if verbose:
        print(f"Starting policy: {policy}")
    
    for iteration in range(max_iterations):
        # Step 1: Policy Evaluation
        V = policy_evaluation(env, policy)
        
        # Step 2: Policy Improvement
        Q = compute_q_values(env, V)
        new_policy = policy_improvement(env, Q)
        
        if verbose:
            print(f"\nIteration {iteration + 1}:")
            print(f"  Policy: {new_policy}")
            print(f"  V-values: {{{', '.join(f'{s}:{V[s]:.1f}' for s in env.states)}}}")
        
        # Check convergence
        if new_policy == policy:
            if verbose:
                print(f"✓ Converged in {iteration + 1} iterations!")
            return new_policy
        
        policy = new_policy
    
    if verbose:
        print(f"Stopped after {max_iterations} iterations (may not have converged)")
    
    return policy


# ============================================================================
# VALUE ITERATION (Direct computation of optimal V)
# ============================================================================

def value_iteration(env: GridWorld,
                   max_iterations: int = 100,
                   theta: float = 1e-4,
                   verbose: bool = True) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    Find optimal value function directly
    
    Algorithm:
        For each state s:
            V*(s) = max_a [R(s,a,s') + γ * V*(s')]
        Extract optimal policy: π*(s) = argmax_a [...]
    """
    
    V = {s: 0.0 for s in env.states}
    
    if verbose:
        print("Value Iteration:")
    
    for iteration in range(max_iterations):
        delta = 0
        
        for state in env.states:
            old_value = V[state]
            
            # Find maximum Q-value over all actions
            max_q_value = -np.inf
            for action in env.get_possible_actions(state):
                next_state, reward = env.get_transitions(state, action)
                q_value = reward + env.gamma * V[next_state]
                max_q_value = max(max_q_value, q_value)
            
            V[state] = max_q_value
            delta = max(delta, abs(old_value - V[state]))
        
        if verbose and iteration % 5 == 0:
            print(f"  Iteration {iteration}: delta={delta:.4f}")
        
        if delta < theta:
            if verbose:
                print(f"  ✓ Converged in {iteration + 1} iterations")
            break
    
    # Extract policy from V
    optimal_policy = {}
    for state in env.states:
        best_action = None
        best_q = -np.inf
        
        for action in env.get_possible_actions(state):
            next_state, reward = env.get_transitions(state, action)
            q_value = reward + env.gamma * V[next_state]
            if q_value > best_q:
                best_q = q_value
                best_action = action
        
        optimal_policy[state] = best_action
    
    return V, optimal_policy


# ============================================================================
# VISUALIZATION
# ============================================================================

def show_policy(policy: Dict[int, str], title: str = "Policy"):
    """Display policy"""
    print(f"\n{title}:")
    print("  State: ", " ".join(str(s) for s in range(5)))
    print("  Action:", " ".join(f"{policy[s][0]:^1s}" for s in range(5)))
    print("          ", " ".join(f"{policy[s][1:]:^1s}" for s in range(5)))


def show_values(V: Dict[int, float], title: str = "Values"):
    """Display value function"""
    print(f"\n{title}:")
    for state in sorted(V.keys()):
        bar_len = max(0, int(V[state] * 2))
        bar = "█" * bar_len if V[state] > 0 else ""
        print(f"  V({state}) = {V[state]:+7.2f} {bar}")


# ============================================================================
# Main: Run examples
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("POLICY IMPROVEMENT & OPTIMAL POLICY")
    print("="*70)
    
    env = GridWorld(gamma=0.9)
    
    # ========== EXAMPLE 1: Manual Policy Improvement ==========
    print("\n1. MANUAL POLICY IMPROVEMENT")
    print("-" * 70)
    
    # Start with a bad policy
    bad_policy = {0: "STAY", 1: "STAY", 2: "STAY", 3: "STAY", 4: "STAY"}
    show_policy(bad_policy, "Initial Policy (bad)")
    
    # Evaluate it
    print("\nEvaluating initial policy...")
    V_bad = policy_evaluation(env, bad_policy)
    show_values(V_bad, "Value function V(s)")
    
    # Compute Q-values
    Q_bad = compute_q_values(env, V_bad)
    print("\nQ-values for each action:")
    for state in range(5):
        print(f"  State {state}:")
        for action in env.get_possible_actions(state):
            q_val = Q_bad[(state, action)]
            marker = " ← best" if action == bad_policy[state] else ""
            print(f"    Q({state}, {action:6s}) = {q_val:+7.2f}{marker}")
    
    # Improve policy
    print("\nImproving policy (pick best action for each state)...")
    improved_policy = policy_improvement(env, Q_bad)
    show_policy(improved_policy, "Improved Policy")
    
    V_improved = policy_evaluation(env, improved_policy)
    show_values(V_improved, "Value function V(s)")
    
    print("\n  ✓ Notice: Values improved!")
    for s in range(5):
        diff = V_improved[s] - V_bad[s]
        if diff != 0:
            print(f"    State {s}: {V_bad[s]:+.1f} → {V_improved[s]:+.1f} (Δ = {diff:+.1f})")
    
    # ========== EXAMPLE 2: Full Policy Iteration ==========
    print("\n" + "="*70)
    print("2. FULL POLICY ITERATION (until convergence)")
    print("-" * 70)
    
    optimal_policy = policy_iteration(env, bad_policy, verbose=True)
    V_optimal = policy_evaluation(env, optimal_policy)
    
    show_policy(optimal_policy, "OPTIMAL POLICY")
    show_values(V_optimal, "Optimal Values")
    
    # ========== EXAMPLE 3: Value Iteration ==========
    print("\n" + "="*70)
    print("3. VALUE ITERATION (direct computation)")
    print("-" * 70)
    
    V_opt, policy_opt = value_iteration(env, verbose=True)
    
    show_policy(policy_opt, "OPTIMAL POLICY (from Value Iteration)")
    show_values(V_opt, "Optimal Values")
    
    # ========== COMPARISON ==========
    print("\n" + "="*70)
    print("4. COMPARISON: Both methods find same optimal policy!")
    print("-" * 70)
    print(f"Policy Iteration result:  {optimal_policy}")
    print(f"Value Iteration result:   {policy_opt}")
    print(f"Are they identical? {optimal_policy == policy_opt}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("• Q(s,a) tells us the value of each action in each state")
    print("• Greedy policy: π(s) = argmax_a Q(s,a)")
    print("• Policy Iteration: Evaluate → Improve → Repeat")
    print("• Value Iteration: Directly compute optimal values")
    print("• Both converge to the same OPTIMAL POLICY")
    print("• Optimal policy: GO RIGHT to reach goal!")
    print("="*70)
