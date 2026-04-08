"""
Example 06: Online Q-Learning vs Gradient Descent

PURPOSE:
--------
Understand WHY Q-Learning is NOT standard Gradient Descent (GD)

KEY INSIGHT FROM LECTURE:
  - Q-Learning Update: φ ← φ - η(Q_φ(s,a) - r - γmax Q_φ(s',a')) ∇_φ Q_φ(s,a)
  - Standard GD would use: ∇_φ y where y = r(s,a) + γmax Q_φ(s',a') (FIXED!)
  - Q-Learning treats y as FIXED (doesn't differentiate through it)
  - This is called "Semi-Gradient" or "Off-Policy" gradient

COMPARISON IN THIS EXAMPLE:
  1. True Gradient Descent: Updates target based on current parameters (unstable!)
  2. Semi-Gradient (Q-Learning): Treats target as fixed (stable)
  3. Both methods on same problem to show the difference

LEARNING OUTCOME:
  ✓ Understand "fixed target" vs "moving target" problem
  ✓ See why Q-Learning is more stable than pure GD
  ✓ Understand gradient computation differences mathematically
  ✓ Learn about semi-gradient methods in deep RL
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


# ============================================================================
# ENVIRONMENT: Simple 1D Grid World
# ============================================================================

class GridWorldEnv:
    """Simple grid world for Q-learning examples"""
    
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.goal = grid_size - 1
        self.current_pos = 0
        
    def reset(self):
        """Reset to start position"""
        self.current_pos = 0
        return self.current_pos
    
    def step(self, action):
        """
        Take action: -1 (LEFT), 0 (STAY), +1 (RIGHT)
        Return: (next_state, reward)
        """
        next_pos = np.clip(self.current_pos + action, 0, self.grid_size - 1)
        self.current_pos = next_pos
        
        # Reward: +10 at goal, -1 per step, -10 for hitting wall
        if next_pos == self.goal:
            reward = 10.0
        elif next_pos == 0 and action == -1:  # Hit left wall
            reward = -10.0
        elif next_pos == self.grid_size - 1 and action == 1:  # Hit right wall
            reward = -10.0
        else:
            reward = -1.0
            
        return next_pos, reward


# ============================================================================
# METHOD 1: SEMI-GRADIENT Q-LEARNING (Correct Implementation)
# ============================================================================

class SemiGradientQLearner:
    """
    Q-Learning with Fixed Target (Semi-Gradient)
    
    Update Rule (from lecture):
      φ ← φ - η(Q_φ(s,a) - [r + γmax Q_φ(s',a')]) ∇_φ Q_φ(s,a)
    
    KEY: Target y = r + γmax Q(s',a') is FIXED
         (we don't differentiate through it)
    """
    
    def __init__(self, state_size=5, action_size=3, lr=0.1, gamma=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr  # Learning rate η
        self.gamma = gamma  # Discount factor γ
        
        # Q-table: dictionary for simplicity
        self.Q = {}
        
    def get_q(self, state, action):
        """Get Q(s,a) value"""
        return self.Q.get((state, action), 0.0)
    
    def get_max_next_q(self, next_state):
        """Get max_a Q(s',a)"""
        q_values = [self.get_q(next_state, a) for a in range(self.action_size)]
        return max(q_values) if q_values else 0.0
    
    def choose_action(self, state, epsilon=0.1):
        """ε-Greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = [self.get_q(state, a) for a in range(self.action_size)]
            return np.argmax(q_values)
    
    def semi_gradient_update(self, state, action, reward, next_state):
        """
        Semi-Gradient Q-Learning Update (CORRECT):
        
        1. Compute current Q-value:  Q_old = Q_φ(s,a)
        2. Compute fixed target:     y = r + γ*max Q_φ(s',a')  ← FIXED!
        3. Compute TD error:         δ = Q_old - y
        4. Compute gradient:         ∇Q = ∇_φ Q_φ(s,a)  ← Only w.r.t. Q
        5. Update:                   φ ← φ - η * δ * ∇Q
        
        In tabular case, ∇Q = 1 (one-hot), so:
           Q(s,a) ← Q(s,a) - η * (Q(s,a) - [r + γ*max Q(s',a')])
        """
        q_old = self.get_q(state, action)
        max_next_q = self.get_max_next_q(next_state)
        
        # FIXED TARGET (key difference!)
        target = reward + self.gamma * max_next_q
        
        # TD error
        td_error = q_old - target
        
        # Update: φ ← φ - η * δ * ∇Q
        # In tabular case: Q ← Q - η * δ
        q_new = q_old - self.lr * td_error
        
        self.Q[(state, action)] = q_new
        
        return td_error, target, q_old, q_new
    
    def train_episode(self, env, epsilon=0.1):
        """Train for one episode"""
        state = env.reset()
        total_reward = 0.0
        
        for _ in range(100):  # Max 100 steps per episode
            action = self.choose_action(state, epsilon)
            next_state, reward = env.step(action)
            
            self.semi_gradient_update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if state == env.goal:
                break
        
        return total_reward


# ============================================================================
# METHOD 2: FULL GRADIENT DESCENT (For Comparison)
# ============================================================================

class FullGradientQLearner:
    """
    Full Gradient Descent Q-Learning (INCORRECT for RL)
    
    This computes gradients through BOTH Q and target:
      ∇_φ [(Q_φ(s,a) - [r + γmax Q_φ(s',a')])²]
    
    Problem: The target ALSO depends on φ!
    Result: Unstable learning, conflicts between current and target Q
    
    This is shown here to demonstrate WHY Q-Learning is different!
    """
    
    def __init__(self, state_size=5, action_size=3, lr=0.1, gamma=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr  # Learning rate η
        self.gamma = gamma  # Discount factor γ
        
        self.Q = {}
        
    def get_q(self, state, action):
        """Get Q(s,a) value"""
        return self.Q.get((state, action), 0.0)
    
    def get_max_next_q(self, next_state):
        """Get max_a Q(s',a)"""
        q_values = [self.get_q(next_state, a) for a in range(self.action_size)]
        return max(q_values) if q_values else 0.0
    
    def choose_action(self, state, epsilon=0.1):
        """ε-Greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = [self.get_q(state, a) for a in range(self.action_size)]
            return np.argmax(q_values)
    
    def full_gradient_update(self, state, action, reward, next_state):
        """
        FULL Gradient Descent Update (PROBLEMATIC):
        
        Standard MSE loss: L = (Q_φ(s,a) - y)²
        where y = r + γ*max Q_φ(s',a')  ← ALSO depends on φ!
        
        Full gradient:
          ∇L = 2(Q_φ(s,a) - y) * ∇_φ[Q_φ(s,a) - y]
              = 2(Q_φ(s,a) - y) * [∇Q_φ(s,a) - ∇y]
              = 2(Q_φ(s,a) - y) * [∇Q_φ(s,a) - γ∇max Q_φ(s',a')]
        
        Update:  φ ← φ - η * ∇L
        
        In tabular form:
          Q(s,a) ← Q(s,a) - lr * 2 * (Q - y) * (1 - γ * 1_max)
        
        Note: The target MOVES as φ changes! (Unstable!)
        """
        q_old = self.get_q(state, action)
        max_next_q = self.get_max_next_q(next_state)
        
        # Target ALSO depends on φ (problematic!)
        target = reward + self.gamma * max_next_q
        
        # Error
        error = q_old - target
        
        # Full gradient through both Q and target
        # In tabular case: gradient through target is also ~1
        # So update includes: (1 - γ) factor from differentiating through target
        gradient_magnitude = 1.0 - self.gamma  # Gradient through target
        
        # Full gradient update (more aggressive)
        update = self.lr * 2 * error * (1.0 + gradient_magnitude)
        q_new = q_old - update
        
        self.Q[(state, action)] = q_new
        
        return error, target, q_old, q_new
    
    def train_episode(self, env, epsilon=0.1):
        """Train for one episode"""
        state = env.reset()
        total_reward = 0.0
        
        for _ in range(100):  # Max 100 steps per episode
            action = self.choose_action(state, epsilon)
            next_state, reward = env.step(action)
            
            self.full_gradient_update(state, action, reward, next_state)
            
            total_reward += reward
            state = next_state
            
            if state == env.goal:
                break
        
        return total_reward


# ============================================================================
# DEMONSTRATION: Compare Both Methods
# ============================================================================

def demonstrate_qlearning_vs_gradient_descent():
    """Show the difference between semi-gradient and full gradient"""
    
    print("=" * 80)
    print("Online Q-Learning vs Gradient Descent")
    print("=" * 80)
    
    print("\n📚 THEORETICAL BACKGROUND:")
    print("-" * 80)
    print("""
Semi-Gradient Q-Learning (CORRECT):
  Update: φ ← φ - η(Q_φ(s,a) - [r + γmax Q_φ(s',a')]) ∇_φ Q_φ(s,a)
  
  Key: Target y = r + γmax Q_φ(s',a') is TREATED AS FIXED
       (we compute gradient ONLY w.r.t. numerator's φ)
  
  Formula breakdown:
    1. TD Error: δ = Q_φ(s,a) - y
    2. Gradient of Q: ∇_φ Q_φ(s,a) = 1 (in tabular)
    3. Update: φ ← φ - η * δ * ∇Q
    
  In tabular Q-table: Q(s,a) ← Q(s,a) - η * δ

Full Gradient Descent (PROBLEMATIC):
  Standard loss: L = (Q_φ(s,a) - y)²
  
  Problem: y ALSO depends on φ!
    y = r + γmax Q_φ(s',a')  ← This has φ in it!
  
  Full gradient would differentiate through BOTH:
    ∇L = 2(Q - y)(∇Q - γ∇max Q)  ← Conflicting signals!
  
  Result: Target keeps moving as we optimize
         Less stable learning, inefficient updates
""")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPARISON")
    print("=" * 80)
    
    # Create environment
    env_1 = GridWorldEnv(grid_size=5)
    env_2 = GridWorldEnv(grid_size=5)
    
    # Create agents
    semi_gradient_agent = SemiGradientQLearner(lr=0.1, gamma=0.9)
    full_gradient_agent = FullGradientQLearner(lr=0.1, gamma=0.9)
    
    # Training
    num_episodes = 200
    semi_gradient_rewards = []
    full_gradient_rewards = []
    
    for episode in range(num_episodes):
        reward_sg = semi_gradient_agent.train_episode(env_1, epsilon=0.1)
        reward_fg = full_gradient_agent.train_episode(env_2, epsilon=0.1)
        
        semi_gradient_rewards.append(reward_sg)
        full_gradient_rewards.append(reward_fg)
        
        if (episode + 1) % 50 == 0:
            avg_sg = np.mean(semi_gradient_rewards[-50:])
            avg_fg = np.mean(full_gradient_rewards[-50:])
            print(f"Episode {episode+1:3d}: Semi-Gradient={avg_sg:7.2f} | "
                  f"Full-Gradient={avg_fg:7.2f}")
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\nFinal Average Rewards (last 50 episodes):")
    print(f"  Semi-Gradient (Q-Learning):  {np.mean(semi_gradient_rewards[-50:]):7.2f}")
    print(f"  Full Gradient:               {np.mean(full_gradient_rewards[-50:]):7.2f}")
    
    print("\nQ-Value Statistics (after training):")
    sg_qvalues = list(semi_gradient_agent.Q.values())
    fg_qvalues = list(full_gradient_agent.Q.values())
    print(f"  Semi-Gradient Q-values:  mean={np.mean(sg_qvalues):.2f}, "
          f"std={np.std(sg_qvalues):.2f}")
    print(f"  Full Gradient Q-values:  mean={np.mean(fg_qvalues):.2f}, "
          f"std={np.std(fg_qvalues):.2f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. SEMI-GRADIENT (Q-Learning) Benefits:
   ✓ Stable learning: Target is fixed, clearer optimization goal
   ✓ Efficient: Focuses only on improving Q-function estimate
   ✓ Gradient-free: Treats target as constant (no second-order effects)
   ✓ Works well in practice: Standard in deep RL (DQN uses target networks)

2. FULL GRADIENT Issues:
   ✗ Unstable: Target moves as we update parameters
   ✗ Conflicting signals: Current Q and target Q fight each other
   ✗ Slow convergence: Gradient computations less efficient
   ✗ Harder to implement: Requires tracking two sets of parameters

3. Why Q-Learning Succeeds in Deep RL:
   • Experience Replay: Breaks correlation in batches
   • Target Networks: Creates true fixed targets (φ_old)
   • Semi-Gradient: Simpler optimization problem
   
   Formula: φ ← φ - η * (Q_φ(s,a) - [r + γmax Q_φ_old(s',a')]) ∇_φ Q_φ(s,a)
           Note: Q_φ_old is FIXED (not updated every step)

4. When Would Full Gradient Help?
   • If target truly didn't depend on φ (supervised learning)
   • If we want to optimize BOTH Q and behavior simultaneously
   • But in RL, separating them (semi-gradient) is better!
""")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Learning curves
    plt.subplot(1, 2, 1)
    window = 10
    sg_smooth = np.convolve(semi_gradient_rewards, np.ones(window)/window, mode='valid')
    fg_smooth = np.convolve(full_gradient_rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(sg_smooth, label='Semi-Gradient Q-Learning', linewidth=2, alpha=0.7)
    plt.plot(fg_smooth, label='Full Gradient Descent', linewidth=2, alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Learning Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of Q-values
    plt.subplot(1, 2, 2)
    plt.hist(sg_qvalues, bins=20, alpha=0.6, label='Semi-Gradient Q-values', edgecolor='black')
    plt.hist(fg_qvalues, bins=20, alpha=0.6, label='Full Gradient Q-values', edgecolor='black')
    plt.xlabel('Q-Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Learned Q-Values')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('qlearning_vs_gradient_descent.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved plot to: qlearning_vs_gradient_descent.png")
    plt.close()
    
    print("\n" + "=" * 80)
    print("LEARNED Q-TABLE (Semi-Gradient Agent)")
    print("=" * 80)
    print("\nQ-values by state (showing first 3 actions):")
    print("State | Action 0 | Action 1 | Action 2 | Max Q | Best Action")
    print("------|----------|----------|----------|-------|------------")
    for state in range(5):
        q0 = semi_gradient_agent.get_q(state, 0)
        q1 = semi_gradient_agent.get_q(state, 1)
        q2 = semi_gradient_agent.get_q(state, 2)
        max_q = max(q0, q1, q2)
        best = ['LEFT', 'STAY', 'RIGHT'][np.argmax([q0, q1, q2])]
        print(f"  {state}   | {q0:8.2f} | {q1:8.2f} | {q2:8.2f} | {max_q:5.2f} | {best}")


if __name__ == "__main__":
    demonstrate_qlearning_vs_gradient_descent()
    
    print("\n" + "=" * 80)
    print("✅ Example Complete!")
    print("=" * 80)
