"""
Example 07: Deep Q-Networks (DQN) for CartPole
With Comprehensive Visualization (Plots + GIFs)

PURPOSE:
--------
Implement a minimal DQN-style agent that highlights epsilon-greedy exploration
and target networks. This version trains online (no replay buffer) to make the
role of exploration easy to see.

CARTPOLE TASK:
  - Observation: [x, x_dot, theta, theta_dot]
    * x: cart position (-2.4 to 2.4)
    * x_dot: cart velocity
    * theta: pole angle (-0.418 to 0.418 radians ≈ ±24°)
    * theta_dot: pole angular velocity
  
  - Actions: [0=LEFT, 1=RIGHT]
  
  - Reward: +1 for each step the pole is upright
  - Episode ends if: pole falls (|theta| > 12°) or cart falls off
  - Goal: Keep pole upright as long as possible (max 500 steps)

KEY INGREDIENTS IN THIS EXAMPLE:
  1. Epsilon-Greedy Exploration: occasionally choose random actions
      → Helps discover better actions early
  
  2. Target Networks: Two networks Q_φ and Q_φ_old
     → Q_φ: Main network, updated every step
     → Q_φ_old: Target network, updated every N steps
     → Implements semi-gradient principle: fixed target = r + γ*max Q_φ_old(s',a')
  
  3. Neural Network: Learn Q-function directly (not tabular)
     → Works for continuous/large state spaces
     → Input: state, Output: Q-values for each action
  
ALGORITHM:
  1. Initialize Q_φ and Q_φ_old with same weights
    2. For each episode:
         - Initialize state s
         - For each step:
             a. Select action: ε-greedy using Q_φ(s,:)
             b. Execute action, observe (s, a, r, s')
             c. Compute target: y = r + γ*max Q_φ_old(s',:)
             d. Update Q_φ: minimize (Q_φ(s,a) - y)²
             e. Update Q_φ_old: every C steps
  3. Track learning progress

REFERENCE:
  Mnih et al., 2015: "Human-level control through deep RL"
  - Original DQN paper that defeated human Atari players
"""

import argparse
import math
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ============================================================================
# CUSTOM CARTPOLE ENVIRONMENT (NO GYM DEPENDENCY)
# ============================================================================

class CartPoleEnv:
    """
    Minimal CartPole environment matching the classic control dynamics.

    Observation: [x, x_dot, theta, theta_dot]
    Actions: 0 = left, 1 = right
    Termination: |x| > 2.4 or |theta| > 12 degrees or max_steps reached
    Reward: 1.0 per step (like Gym CartPole-v1)
    """

    def __init__(self, max_steps: int = 500, seed: int | None = None) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.x_threshold = 2.4
        self.theta_threshold_radians = 12.0 * math.pi / 180.0

        self.max_steps = max_steps
        self.np_random = np.random.RandomState(seed)
        self.state: np.ndarray | None = None
        self.steps = 0

    def seed(self, seed: int | None = None) -> None:
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.seed(seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.state is not None, "Call reset() before step()."
        assert action in (0, 1), "Action must be 0 (left) or 1 (right)."

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (
            self.gravity * sintheta - costheta * temp
        ) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        truncated = self.steps >= self.max_steps
        reward = 1.0

        return self.state.copy(), reward, terminated, truncated, {}

    @property
    def observation_space_shape(self) -> tuple[int]:
        return (4,)

    @property
    def action_space_n(self) -> int:
        return 2


# ============================================================================
# NEURAL NETWORK: Q-FUNCTION APPROXIMATOR
# ============================================================================

class QNetwork(nn.Module):
    """
    Neural network to approximate Q(s,a)
    
    Architecture:
      Input: State (4 dimensions for CartPole)
        ↓
      Hidden layer 1: 128 units + ReLU
        ↓
      Hidden layer 2: 128 units + ReLU
        ↓
      Output: Q-values for each action
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        Forward pass: state → Q-values
        
        Args:
            state: shape (batch_size, state_dim)
        
        Returns:
            q_values: shape (batch_size, action_dim)
                     q_values[i, a] = Q(state[i], action=a)
        """
        return self.net(state)


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    """
    Deep Q-Network Agent
    
    Key Components:
      1. Q_network: Main network (updated every step)
      2. target_network: Target network (updated every C steps)
      3. optimizer: Adam optimizer for gradient descent
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        target_update_freq=1000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        
        # Device: use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks: Q_network and target_network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Step counter (for target network updates)
        self.step_count = 0
        
        # Training history
        self.loss_history = []
        self.reward_history = []
    
    def select_action(self, state):
        """
        ε-Greedy action selection
        
        With probability ε: Random action (exploration)
        With probability 1-ε: Best action according to Q_network (exploitation)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def train_step(self, state, action, reward, next_state, done):
        """
        One gradient descent step on a single transition
        
        DQN Loss:
          L = E[(Q_φ(s,a) - y)²]
          where y = r + γ * max_a' Q_φ_old(s', a')  ← Fixed target!
        
        Update:
          φ ← φ - ∇_φ L  (standard gradient descent)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([done]).to(self.device)
        
        # Current Q-value: Q_φ(s,a)
        q_current = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Target Q-value: y = r + γ*max Q_φ_old(s',a')
        with torch.no_grad():  # ← STOP GRADIENT on target (semi-gradient principle!)
            q_next_target = self.target_network(next_state_tensor).max(dim=1)[0]
            q_target = reward_tensor + self.gamma * q_next_target * (1 - done_tensor)
        
        # MSE Loss: (Q_current - y)²
        loss = nn.functional.mse_loss(q_current, q_target)
        
        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update step counter and target network if needed
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"  Target network updated at step {self.step_count}")
        
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_dqn(num_episodes=500, render=False, save_dir=None):
    """
    Train DQN agent on CartPole
    
    CartPole-v1:
      - Observation space: 4D continuous
      - Action space: 2 discrete actions
      - Goal: 500 consecutive steps without falling
    """
    
    if save_dir is None:
        save_dir = Path("outputs")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Deep Q-Network (DQN) Training on CartPole")
    print("=" * 80)
    print(f"\nOutput directory: {save_dir.resolve()}")
    
    # Create environment
    env = CartPoleEnv(max_steps=500)
    
    # Agent hyperparameters
    state_dim = env.observation_space_shape[0]
    action_dim = env.action_space_n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        target_update_freq=1000
    )
    
    print(f"\nEnvironment:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"\nAgent Configuration:")
    print(f"  Learning rate: {agent.learning_rate}")
    print(f"  Gamma (discount): {agent.gamma}")
    print(f"  Target Update Frequency: {agent.target_update_freq} steps")
    
    print(f"\n{'Episode':<10} {'Reward':<10} {'Avg Reward':<15} {'Epsilon':<10}")
    print("-" * 80)
    
    # Training loop
    best_reward = 0
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Online TD update (no replay buffer)
            agent.train_step(state, action, reward, next_state, float(done))
            
            episode_reward += reward
            state = next_state
        
        # Update exploration rate
        agent.update_epsilon()
        agent.reward_history.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(agent.reward_history[-50:])
            best_reward = max(best_reward, avg_reward)
            print(f"{episode+1:<10} {episode_reward:<10.1f} {avg_reward:<15.1f} "
                f"{agent.epsilon:<10.4f}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    # Results
    final_avg = np.mean(agent.reward_history[-100:])
    print(f"\nFinal Average Reward (last 100 episodes): {final_avg:.1f}")
    print(f"Best Average Reward: {best_reward:.1f}")
    print(f"Max Reward (CartPole-v1 limit): 500.0")
    
    if final_avg >= 450:
        print("\n✅ SUCCESS! Agent learned to balance pole effectively!")
    elif final_avg >= 300:
        print("\n⚠️ GOOD! Agent learned decent policy but not perfect")
    else:
        print("\n❌ Agent needs more training")
    
    return agent, save_dir


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(save_dir: Path, agent):
    """
    Plot comprehensive training metrics
    Similar to diffusion_cartpole_study.py plotting style
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Episode rewards
    episodes = range(1, len(agent.reward_history) + 1)
    window = 50
    if len(agent.reward_history) > window:
        moving_avg = np.convolve(agent.reward_history, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(episodes, agent.reward_history, alpha=0.3, color='#0072B2', linewidth=1)
        axes[0, 0].plot(range(window, len(agent.reward_history) + 1), moving_avg, 
                       color='#D55E00', linewidth=2.5, label=f'Moving Avg ({window})')
    axes[0, 0].axhline(y=500, color='#009E73', linestyle='--', linewidth=2, label='Target (500)')
    axes[0, 0].set_xlabel('Episode', fontsize=11)
    axes[0, 0].set_ylabel('Reward', fontsize=11)
    axes[0, 0].set_title('Episode Rewards Over Training', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.25)
    
    # Plot 2: Training loss
    if len(agent.loss_history) > 100:
        window_loss = 100
        loss_moving_avg = np.convolve(agent.loss_history, np.ones(window_loss)/window_loss, mode='valid')
        loss_steps = np.arange(0, len(agent.loss_history), 10)[:len(agent.loss_history[::10])]
        axes[0, 1].semilogy(loss_steps, agent.loss_history[::10], alpha=0.3, color='#0072B2', linewidth=1)
        axes[0, 1].semilogy(range(0, len(loss_moving_avg) * 10, 10), loss_moving_avg, 
                           color='#D55E00', linewidth=2.5, label='Moving Avg (100)')
        axes[0, 1].set_xlabel('Training Step', fontsize=11)
        axes[0, 1].set_ylabel('MSE Loss (log scale)', fontsize=11)
        axes[0, 1].set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.25, which='both')
    
    # Plot 3: Epsilon decay
    epsilon_history = []
    eps = 1.0
    for _ in range(len(agent.reward_history)):
        epsilon_history.append(eps)
        eps = max(agent.epsilon_min, eps * agent.epsilon_decay)
    
    axes[1, 0].plot(epsilon_history, color='#0072B2', linewidth=2)
    axes[1, 0].fill_between(range(len(epsilon_history)), epsilon_history, alpha=0.3, color='#0072B2')
    axes[1, 0].set_xlabel('Episode', fontsize=11)
    axes[1, 0].set_ylabel('Exploration Rate (ε)', fontsize=11)
    axes[1, 0].set_title('Epsilon Decay Schedule', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.05])
    axes[1, 0].grid(True, alpha=0.25)
    
    # Plot 4: Cumulative rewards
    cumulative_rewards = np.cumsum(agent.reward_history)
    axes[1, 1].plot(cumulative_rewards, color='#009E73', linewidth=2, alpha=0.7)
    axes[1, 1].fill_between(range(len(cumulative_rewards)), cumulative_rewards, alpha=0.2, color='#009E73')
    axes[1, 1].set_xlabel('Episode', fontsize=11)
    axes[1, 1].set_ylabel('Cumulative Reward', fontsize=11)
    axes[1, 1].set_title('Cumulative Reward Progress', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.25)
    
    fig.tight_layout()
    out_file = save_dir / "dqn_training_curves_cartpole.png"
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_file}")


def rollout_trajectory(env, agent, num_steps=500):
    """
    Rollout one trajectory from random initial state
    Returns trajectory data for visualization
    """
    state, _ = env.reset()
    
    trajectory = {
        'x': [state[0]],
        'x_dot': [state[1]],
        'theta': [state[2]],
        'theta_dot': [state[3]],
        'actions': []
    }
    
    episode_reward = 0
    for _ in range(num_steps):
        # Use deterministic policy (epsilon=0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q_values = agent.q_network(state_tensor)
        action = q_values.argmax(dim=1).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        trajectory['x'].append(next_state[0])
        trajectory['x_dot'].append(next_state[1])
        trajectory['theta'].append(next_state[2])
        trajectory['theta_dot'].append(next_state[3])
        trajectory['actions'].append(action)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    return trajectory, episode_reward


def plot_trajectory(save_dir: Path, traj, dt=0.02):
    """
    Plot state trajectories with professional formatting
    """
    t = [i * dt for i in range(len(traj['theta']))]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # Pole angle
    axes[0, 0].plot(t, np.degrees(traj['theta']), color='#D55E00', linewidth=2)
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 0].set_ylabel('Angle (degrees)', fontsize=11)
    axes[0, 0].set_title('Pole Angle (θ)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].fill_between(t, np.degrees(traj['theta']), alpha=0.2, color='#D55E00')
    
    # Pole angular velocity
    axes[0, 1].plot(t, traj['theta_dot'], color='#0072B2', linewidth=2)
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=11)
    axes[0, 1].set_title('Pole Angular Velocity (θ̇)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.25)
    
    # Cart position
    axes[1, 0].plot(t, traj['x'], color='#009E73', linewidth=2)
    axes[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[1, 0].set_ylabel('Position (m)', fontsize=11)
    axes[1, 0].set_title('Cart Position (x)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].fill_between(t, traj['x'], alpha=0.2, color='#009E73')
    
    # Cart velocity
    axes[1, 1].plot(t, traj['x_dot'], color='#CC79A7', linewidth=2)
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('Velocity (m/s)', fontsize=11)
    axes[1, 1].set_title('Cart Velocity (ẋ)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.25)
    
    fig.suptitle('DQN CartPole - Learned Behavior Trajectory', fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    
    out_file = save_dir / "dqn_trajectory_cartpole.png"
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_file}")


def create_cartpole_animation(save_dir: Path, trajectory, episode_reward):
    """
    Create GIF animation of CartPole balancing
    Shows visual representation like in diffusion_cartpole_study
    """
    if not HAS_IMAGEIO:
        print("⚠️  imageio not installed, skipping GIF generation")
        return
    
    print("\n🎬 Creating CartPole animation...")
    frame_paths = []
    pole_len = 1.0
    cart_width = 0.4
    cart_height = 0.3
    
    # Sample frames to keep file size reasonable
    step_size = max(1, len(trajectory['theta']) // 100)
    
    for i in range(0, len(trajectory['theta']), step_size):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = trajectory['x'][i]
        theta = trajectory['theta'][i]
        
        # Draw ground
        ax.plot([-3, 3], [0, 0], 'k-', linewidth=3)
        ax.fill_between([-3, 3], -0.1, 0, color='gray', alpha=0.3)
        
        # Draw cart
        cart_x = [x - cart_width/2, x + cart_width/2, 
                  x + cart_width/2, x - cart_width/2, x - cart_width/2]
        cart_y = [0, 0, cart_height, cart_height, 0]
        ax.plot(cart_x, cart_y, 'b-', linewidth=2)
        ax.fill(cart_x, cart_y, color='#0072B2', alpha=0.5)
        
        # Draw wheels
        wheel_radius = 0.1
        wheel1_x = x - cart_width/3
        wheel2_x = x + cart_width/3
        circle1 = plt.Circle((wheel1_x, -wheel_radius), wheel_radius, color='black')
        circle2 = plt.Circle((wheel2_x, -wheel_radius), wheel_radius, color='black')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        # Draw pole
        pole_x = x + pole_len * np.sin(theta)
        pole_y = cart_height + pole_len * np.cos(theta)
        ax.plot([x, pole_x], [cart_height, pole_y], 'r-', linewidth=3)
        
        # Draw pole tip
        circle_pole = plt.Circle((pole_x, pole_y), 0.1, color='#D55E00', alpha=0.7)
        ax.add_patch(circle_pole)
        
        # Formatting
        ax.set_xlim([-3, 3])
        ax.set_ylim([-0.5, 2.5])
        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)', fontsize=11)
        ax.set_ylabel('Height (m)', fontsize=11)
        ax.set_title(f'DQN CartPole Balancing - Step {i:04d} | '
                    f'θ={np.degrees(theta):6.1f}° | x={x:6.2f}m',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25)
        
        frame_path = save_dir / f"frame_{i:04d}.png"
        fig.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)
    
    # Create GIF
    if frame_paths:
        gif_path = save_dir / "dqn_cartpole_balance.gif"
        images = [imageio.imread(str(path)) for path in frame_paths]
        imageio.mimsave(str(gif_path), images, duration=0.05)
        print(f"✅ Saved: {gif_path}")
        
        # Clean up frames
        for path in frame_paths:
            path.unlink()





# ============================================================================
# EVALUATION: TEST TRAINED AGENT
# ============================================================================

def evaluate_agent(agent, num_episodes=10, render=False):
    """
    Test trained agent on CartPole
    
    Note: Use epsilon=0 (no exploration, only exploitation)
    """
    
    print("\n" + "=" * 80)
    print("EVALUATING TRAINED AGENT")
    print("=" * 80)
    
    env = CartPoleEnv(max_steps=500)
    
    # Temporarily disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    test_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        test_rewards.append(episode_reward)
        print(f"  Test Episode {episode+1:2d}: Reward = {episode_reward:7.1f}")
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    avg_test_reward = np.mean(test_rewards)
    std_test_reward = np.std(test_rewards)
    print(f"\nAverage Test Reward: {avg_test_reward:.1f} ± {std_test_reward:.1f}")
    print(f"Max Test Reward: {max(test_rewards):.1f}")
    print(f"Min Test Reward: {min(test_rewards):.1f}")
    
    return test_rewards


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deep Q-Network (DQN) for CartPole")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate for optimizer")
    parser.add_argument("--episodes-eval", type=int, default=5,
                       help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\n" + "=" * 80)
    print("DEEP Q-NETWORK (DQN) FOR CARTPOLE")
    print("With Comprehensive Visualization (Plots + GIFs)")
    print("=" * 80)
    
    # Training
    print("\n[1/4] Training DQN agent...")
    agent, save_dir = train_dqn(num_episodes=args.episodes, render=False, save_dir="outputs")
    
    # Plotting training curves
    print("\n[2/4] Generating training visualization...")
    plot_training_curves(save_dir, agent)
    
    # Rollout and visualize trajectory
    print("\n[3/4] Collecting learned behavior trajectory...")
    env = CartPoleEnv(max_steps=500)
    agent.epsilon = 0  # Deterministic policy
    traj, traj_reward = rollout_trajectory(env, agent, num_steps=500)
    plot_trajectory(save_dir, traj)
    print(f"  Trajectory reward: {traj_reward:.1f}")
    
    # Create animation
    print("\n[4/4] Creating CartPole balance animation...")
    create_cartpole_animation(save_dir, traj, traj_reward)
    
    # Evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    test_rewards = evaluate_agent(agent, num_episodes=args.episodes_eval, render=False)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"""
Summary:
  - Training episodes: {args.episodes}
  - Final average reward: {np.mean(agent.reward_history[-100:]):.1f}
  - Test average reward: {np.mean(test_rewards):.1f}
  - Output directory: {save_dir.resolve()}
  
Generated files:
  - dqn_training_curves_cartpole.png: Training metrics
  - dqn_trajectory_cartpole.png: Learned trajectory
  - dqn_cartpole_balance.gif: CartPole animation

Key Innovations Used:
    1. ε-Greedy: Exploration-exploitation tradeoff
    2. Target Networks: Fixed targets (semi-gradient principle)
    3. Neural Networks: Q-function approximation

Connection to Previous Examples:
  • Example 1-3: Tabular RL (small state spaces)
  • Example 4: Online Q-Learning (Q-table)
  • Example 5: Offline Learning (Fitted Q Iteration)
  • Example 6: Semi-Gradient Principle ← KEY for DQN!
  • Example 7: DQN ← This example!
    """)





if __name__ == "__main__":
    main()
