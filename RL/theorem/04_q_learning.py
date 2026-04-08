"""
Simple RL Example 4: Q-Learning (Model-Free Learning)
======================================================
Learning optimal policy WITHOUT knowing the environment model

In previous examples, we assumed we KNOW:
- State transitions: What state you go to from each action
- Rewards: How much reward you get

In Q-Learning:
- We DON'T know the model
- We LEARN Q-values through interaction and trial-and-error
- Update rule: Q(s,a) <- Q(s,a) + alpha[R + gamma*max Q(s',a') - Q(s,a)]
"""

import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class GridWorld:
    """1D Grid World: States 0-4, Goal at 4"""
    
    def __init__(self):
        self.states = list(range(5))
        self.goal = 4
        self.current_state = None
    
    def reset(self) -> int:
        """Reset to random starting state"""
        self.current_state = np.random.randint(0, 5)
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float]:
        """
        Take action (0=LEFT, 1=STAY, 2=RIGHT)
        Returns: (next_state, reward)
        """
        action_map = {0: -1, 1: 0, 2: +1}
        movement = action_map[action]
        next_state = self.current_state + movement
        
        # Handle bounds
        if next_state < 0 or next_state > 4:
            reward = -10  # Penalty for hitting wall
            next_state = self.current_state  # Stay in place
        elif next_state == self.goal:
            reward = +10  # Goal reward
        else:
            reward = -1  # Step cost
        
        self.current_state = next_state
        return next_state, reward


class QLearningAgent:
    """
    Agent that learns Q-values through interaction
    
    Q-Learning Update Rule:
        Q(s,a) <- Q(s,a) + alpha[r + gamma*max Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, 
                 num_states: int = 5,
                 num_actions: int = 3,
                 alpha: float = 0.1,      # Learning rate
                 gamma: float = 0.9,      # Discount factor
                 epsilon: float = 0.1):   # Exploration rate
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table: Q(s,a)
        self.Q = np.zeros((num_states, num_actions))
        
        # Track learning history
        self.episode_rewards = []
        self.episode_steps = []
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """Choose action using epsilon-greedy strategy"""
        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.num_actions)
        else:
            # Greedy action (exploitation)
            return np.argmax(self.Q[state, :])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """Q-Learning update step"""
        
        # Current Q-value
        current_q = self.Q[state, action]
        
        # Maximum Q-value in next state
        max_next_q = np.max(self.Q[next_state, :]) if not done else 0
        
        # Target: what we think Q should be
        target = reward + self.gamma * max_next_q
        
        # Update: move towards target
        self.Q[state, action] = current_q + self.alpha * (target - current_q)
    
    def train_episode(self, env: GridWorld, max_steps: int = 20) -> Tuple[float, int]:
        """Run one training episode"""
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Choose action
            action = self.choose_action(state, training=True)
            
            # Take action in environment
            next_state, reward = env.step(action)
            total_reward += reward
            steps += 1
            
            # Check if done
            done = next_state == env.goal
            
            # Q-Learning update
            self.update(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(steps)
        
        return total_reward, steps
    
    def get_policy(self) -> Dict[int, int]:
        """Extract learned policy from Q-table"""
        policy = {}
        for state in range(self.num_states):
            policy[state] = np.argmax(self.Q[state, :])
        return policy
    
    def show_q_table(self):
        """Display learned Q-values"""
        action_names = ["LEFT", "STAY", "RIGHT"]
        print("\nLearned Q-table:")
        print(f"{'State':^8} | " + " | ".join(f"{act:^10s}" for act in action_names))
        print("-" * 50)
        
        for state in range(self.num_states):
            q_vals = [f"{self.Q[state, a]:+7.2f}" for a in range(self.num_actions)]
            best_action = action_names[np.argmax(self.Q[state, :])]
            print(f"{state:^8} | " + " | ".join(f"{q:^10s}" for q in q_vals) + f" <- {best_action}")


def train_q_learning(env: GridWorld, 
                    num_episodes: int = 100,
                    verbose: bool = True) -> QLearningAgent:
    """Train Q-Learning agent"""
    
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    
    if verbose:
        print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        reward, steps = agent.train_episode(env)
        
        if verbose and (episode + 1) % 20 == 0:
            avg_reward = np.mean(agent.episode_rewards[-20:])
            avg_steps = np.mean(agent.episode_steps[-20:])
            print(f"  Episode {episode + 1}: Avg reward = {avg_reward:+.1f}, Avg steps = {avg_steps:.1f}")
    
    return agent


def plot_learning_curves(agent: QLearningAgent):
    """Plot learning progress"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    window = 10
    avg_rewards = np.convolve(agent.episode_rewards, 
                             np.ones(window)/window, 
                             mode='valid')
    axes[0].plot(agent.episode_rewards, alpha=0.3, label='Episode reward')
    axes[0].plot(range(window-1, len(agent.episode_rewards)), avg_rewards, 
                 linewidth=2, label=f'Avg ({window} episodes)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Learning Progress: Reward')
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # Plot steps
    avg_steps = np.convolve(agent.episode_steps, 
                           np.ones(window)/window, 
                           mode='valid')
    axes[1].plot(agent.episode_steps, alpha=0.3, label='Episode steps')
    axes[1].plot(range(window-1, len(agent.episode_steps)), avg_steps,
                linewidth=2, label=f'Avg ({window} episodes)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Complete')
    axes[1].set_title('Learning Progress: Efficiency')
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("="*70)
    print("Q-LEARNING: Learning from Interaction")
    print("="*70)
    
    env = GridWorld()
    
    # ========== EXAMPLE 1: Before Training ==========
    print("\n1. BEFORE TRAINING")
    print("-" * 70)
    print("Random Q-table (all zeros initially):")
    agent_untrained = QLearningAgent()
    agent_untrained.show_q_table()
    
    print("\nIf we follow random actions:")
    random_rewards = []
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        for step in range(20):
            action = np.random.randint(0, 3)  # Random action
            next_state, reward = env.step(action)
            total_reward += reward
            if next_state == env.goal:
                break
            state = next_state
        random_rewards.append(total_reward)
    
    print(f"  Average reward (random policy): {np.mean(random_rewards):.1f}")
    
    # ========== EXAMPLE 2: Training Progress ==========
    print("\n" + "="*70)
    print("2. TRAINING PROCESS")
    print("-" * 70)
    
    agent = train_q_learning(env, num_episodes=200, verbose=True)
    
    # ========== EXAMPLE 3: After Training ==========
    print("\n" + "="*70)
    print("3. AFTER TRAINING")
    print("-" * 70)
    print("Learned Q-table:")
    agent.show_q_table()
    
    # ========== EXAMPLE 4: Test Learned Policy ==========
    print("\n" + "="*70)
    print("4. TEST LEARNED POLICY")
    print("-" * 70)
    
    print("\nTesting learned policy (no exploration):")
    test_rewards = []
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        action_sequence = []
        action_map = {0: "LEFT", 1: "STAY", 2: "RIGHT"}
        
        for step in range(20):
            action = agent.choose_action(state, training=False)  # Greedy only
            next_state, reward = env.step(action)
            action_sequence.append(action_map[action])
            total_reward += reward
            
            if next_state == env.goal:
                break
            state = next_state
        
        test_rewards.append(total_reward)
        if episode == 0:
            print(f"  Sample trajectory: {' -> '.join(action_sequence)}")
    
    print(f"  Average reward (learned policy): {np.mean(test_rewards):.1f}")
    print(f"  Success rate: {sum(1 for r in test_rewards if r >= 0) * 100 / len(test_rewards):.0f}%")
    
    # ========== EXAMPLE 5: Visualization ==========
    print("\n" + "="*70)
    print("5. LEARNING CURVES")
    print("-" * 70)
    
    fig = plot_learning_curves(agent)
    print("\n  Saved plots to 'rl_learning_curves.png'")
    fig.savefig('/home/quangvd7/self_learning/Robot_learning/RL/rl_learning_curves.png', dpi=150)
    plt.close()
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("• Q-Learning learns from INTERACTION, not from model")
    print("• Q-values improve with experience (bootstrapping)")
    print("• Epsilon-greedy: balance EXPLORATION vs EXPLOITATION")
    print("• Q(s,a) <- Q(s,a) + alpha[r + gamma*max Q(s',a') - Q(s,a)]")
    print("• Works for problems where we don't know the model")
    print("="*70)
