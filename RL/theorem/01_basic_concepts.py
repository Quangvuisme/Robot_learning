"""
Simple RL Example 1: Basic Concepts
====================================
Understanding: State, Action, Reward, and Policy

In reinforcement learning:
- STATE (s): Description of current situation in environment
- ACTION (a): What agent can do from a state
- REWARD (r): Numerical feedback from environment
- POLICY (π): Rule that agent follows to choose actions
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================================
# EXAMPLE 1: Simple Grid World
# ============================================================================
"""
Imagine a 1D world with 5 positions:
    0 ---- 1 ---- 2 ---- 3 ---- 4
    START              GOAL(+10)

Agent starts at position 0.
Actions: LEFT (move -1), STAY (0), RIGHT (move +1)
Goal: Reach position 4 (get reward of +10)
Each step costs -1 energy
"""

class SimpleGridWorld:
    """1D grid world environment"""
    
    def __init__(self):
        self.states = [0, 1, 2, 3, 4]
        self.actions = ["LEFT", "STAY", "RIGHT"]
        self.action_map = {"LEFT": -1, "STAY": 0, "RIGHT": 1}
        self.current_state = 0
        self.goal_state = 4
        
    def reset(self):
        """Reset agent to start"""
        self.current_state = 0
        return self.current_state
    
    def step(self, action: str) -> Tuple[int, float, bool]:
        """
        Execute action and return: (next_state, reward, done)
        
        Reward: +10 at goal, -1 per step (cost), -10 if out of bounds
        """
        # Get next state
        next_state = self.current_state + self.action_map[action]
        
        # Check bounds
        done = False
        if next_state < 0 or next_state > 4:
            reward = -10  # Penalty for hitting wall
            next_state = self.current_state  # Stay in place
        elif next_state == self.goal_state:
            reward = +10  # Reward for reaching goal
            done = True
        else:
            reward = -1  # Step cost
        
        self.current_state = next_state
        return next_state, reward, done
    
    def print_state(self):
        """Visualize current state"""
        grid = ["." for _ in self.states]
        grid[self.current_state] = "X"
        grid[self.goal_state] = "G"
        print("Grid: " + " ".join(grid))
        print(f"Agent at position {self.current_state}")


# ============================================================================
# Example 2: POLICY (Rule for choosing actions)
# ============================================================================

class Policy:
    """Base policy class"""
    
    def __init__(self, env: SimpleGridWorld):
        self.env = env
    
    def get_action(self, state: int) -> str:
        """Given a state, return which action to take"""
        raise NotImplementedError


class AlwaysRightPolicy(Policy):
    """Simple policy: always move RIGHT"""
    
    def get_action(self, state: int) -> str:
        return "RIGHT"


class SmartPolicy(Policy):
    """Smart policy: move right to reach goal, but don't hit walls"""
    
    def get_action(self, state: int) -> str:
        if state < 4:
            return "RIGHT"  # Move towards goal
        else:
            return "STAY"   # Already at goal


class RandomPolicy(Policy):
    """Random policy: choose action uniformly at random"""
    
    def get_action(self, state: int) -> str:
        return np.random.choice(self.env.actions)


# ============================================================================
# Example 3: TRAJECTORY (sequence of states and rewards)
# ============================================================================

def run_episode(env: SimpleGridWorld, policy: Policy, max_steps: int = 20):
    """
    Run one episode: agent interacts with environment following policy
    
    Returns trajectory: list of (state, action, reward) tuples
    """
    state = env.reset()
    trajectory = []
    total_reward = 0
    
    for step in range(max_steps):
        action = policy.get_action(state)
        next_state, reward, done = env.step(action)
        
        trajectory.append((state, action, reward, next_state))
        total_reward += reward
        
        if done:
            break
        
        state = next_state
    
    return trajectory, total_reward


# ============================================================================
# Main: Run examples
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RL BASICS: State, Action, Reward, Policy")
    print("="*70)
    
    env = SimpleGridWorld()
    
    # Show environment
    print("\n1. ENVIRONMENT SETUP")
    print("-" * 70)
    print("States:", env.states)
    print("Actions:", env.actions)
    env.print_state()
    
    # Demo: Manual interaction
    print("\n2. MANUAL INTERACTION (what happens when we take actions)")
    print("-" * 70)
    state = env.reset()
    print(f"Start at state: {state}")
    
    for action in ["RIGHT", "RIGHT", "STAY"]:
        next_state, reward, done = env.step(action)
        print(f"  Action: {action:6s} → Next state: {next_state}, Reward: {reward:+3.0f}, Done: {done}")
    
    # Compare policies
    print("\n3. COMPARE THREE DIFFERENT POLICIES")
    print("-" * 70)
    
    policies = [
        ("Always Right", AlwaysRightPolicy(env)),
        ("Smart (move right)", SmartPolicy(env)),
        ("Random", RandomPolicy(env))
    ]
    
    num_episodes = 10
    
    for policy_name, policy in policies:
        print(f"\n  Policy: {policy_name}")
        total_rewards = []
        
        for episode in range(num_episodes):
            trajectory, total_reward = run_episode(env, policy, max_steps=20)
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        print(f"    Average reward over {num_episodes} episodes: {avg_reward:+.2f}")
        print(f"    Sample trajectory (first 5 steps):")
        
        # Show one trajectory
        trajectory, _ = run_episode(env, policy, max_steps=20)
        for i, (s, a, r, s_next) in enumerate(trajectory[:5]):
            print(f"      Step {i}: State {s} --[{a}]--> State {s_next}, Reward {r:+3.0f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("• POLICY determines which ACTION agent takes in each STATE")
    print("• Different policies lead to different TRAJECTORIES and REWARDS")
    print("• Goal: Find OPTIMAL policy that maximizes cumulative reward!")
    print("="*70)
