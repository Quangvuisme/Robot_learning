"""
Simple RL Example 3.5 (Alternative): Fitted Q Iteration with Lookup Table
============================================================================
A clearer version of FQI that works reliably on the grid world

This version uses a lookup table (dictionary) instead of neural network
to make the learning dynamics clear and show how FQI works.
"""

import numpy as np
from typing import Dict, Tuple, List


class GridWorld:
    """1D Grid World: States 0-4, Goal at 4"""
    
    def __init__(self):
        self.states = list(range(5))
        self.goal = 4
        self.current_state = None
    
    def reset(self) -> int:
        self.current_state = np.random.randint(0, 5)
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float]:
        action_map = {0: -1, 1: 0, 2: +1}
        movement = action_map[action]
        next_state = self.current_state + movement
        
        if next_state < 0 or next_state > 4:
            reward = -10
            next_state = self.current_state
        elif next_state == self.goal:
            reward = +10
        else:
            reward = -1
        
        self.current_state = next_state
        return next_state, reward


# ============================================================================
# DATASET COLLECTION
# ============================================================================

def collect_dataset_random(env: GridWorld, 
                          num_samples: int = 300,
                          seed: int = 42) -> List[Tuple]:
    """Collect dataset using random exploration"""
    np.random.seed(seed)
    dataset = []
    
    for _ in range(num_samples):
        state = env.reset()
        action = np.random.randint(0, 3)
        next_state, reward = env.step(action)
        dataset.append((state, action, reward, next_state))
    
    return dataset


def collect_dataset_mixed(env: GridWorld,
                         num_samples: int = 300,
                         seed: int = 42) -> List[Tuple]:
    """
    Collect dataset using mixed policy:
    - 50% random exploration
    - 50% go-right policy
    This gives better coverage of good trajectories
    """
    np.random.seed(seed)
    dataset = []
    
    for i in range(num_samples):
        state = env.reset()
        
        # Choose exploration strategy
        if i % 2 == 0:
            # Random
            action = np.random.randint(0, 3)
        else:
            # Go right (better policy)
            action = 2  # RIGHT
        
        next_state, reward = env.step(action)
        dataset.append((state, action, reward, next_state))
    
    return dataset


# ============================================================================
# FITTED Q ITERATION WITH LOOKUP TABLE
# ============================================================================

class FittedQTable:
    """
    Simple lookup table Q-function approximator
    Q(s,a) stored as dictionary
    """
    
    def __init__(self):
        # Initialize Q-table
        self.Q = {}
        for s in range(5):
            for a in range(3):
                self.Q[(s, a)] = 0.0
    
    def predict(self, state: int, action: int) -> float:
        """Get Q-value"""
        return self.Q.get((state, action), 0.0)
    
    def update(self, state: int, action: int, target: float, lr: float = 0.1):
        """Update Q-value towards target"""
        old_q = self.Q[(state, action)]
        self.Q[(state, action)] = old_q + lr * (target - old_q)
    
    def get_max_next_q(self, next_state: int) -> float:
        """Get max Q-value for next state"""
        max_q = -np.inf
        for a in range(3):
            max_q = max(max_q, self.Q[(next_state, a)])
        return max_q if max_q > -np.inf else 0.0


def fitted_q_iteration_table(dataset: List[Tuple],
                             gamma: float = 0.9,
                             iterations: int = 30,
                             lr: float = 0.1,
                             verbose: bool = True) -> Tuple[FittedQTable, List[float]]:
    """
    Fitted Q Iteration with lookup table
    
    Algorithm:
    Repeat for N iterations:
      For each sample (s, a, r, s') in dataset:
        y_i <- r + γ * max_a' Q(s', a')
        Q(s, a) <- Q(s, a) + lr * (y_i - Q(s, a))
    """
    
    Q_table = FittedQTable()
    loss_history = []
    
    if verbose:
        print("Fitted Q Iteration running...")
    
    for iteration in range(iterations):
        total_loss = 0
        
        # Shuffle dataset
        shuffled = dataset.copy()
        np.random.shuffle(shuffled)
        
        # Process each sample
        for state, action, reward, next_state in shuffled:
            # Compute target
            max_next_q = Q_table.get_max_next_q(next_state)
            target = reward + gamma * max_next_q
            
            # Current prediction
            current_q = Q_table.predict(state, action)
            
            # Loss
            loss = (target - current_q) ** 2
            total_loss += loss
            
            # Update
            Q_table.update(state, action, target, lr=lr)
        
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: Average Loss = {avg_loss:.6f}")
    
    return Q_table, loss_history


# ============================================================================
# POLICY EXTRACTION & EVALUATION
# ============================================================================

def get_policy(Q_table: FittedQTable) -> Dict[int, int]:
    """Extract greedy policy from Q-table"""
    policy = {}
    for state in range(5):
        best_action = 0
        best_q = Q_table.predict(state, 0)
        for action in range(1, 3):
            q = Q_table.predict(state, action)
            if q > best_q:
                best_q = q
                best_action = action
        policy[state] = best_action
    return policy


def test_policy(env: GridWorld, policy: Dict[int, int], num_episodes: int = 50) -> Tuple:
    """Evaluate policy"""
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(20):
            action = policy[state]
            next_state, reward = env.step(action)
            total_reward += reward
            if next_state == env.goal:
                break
            state = next_state
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)


def show_q_table(Q_table: FittedQTable):
    """Display Q-table"""
    print("\nLearned Q-values:")
    print(f"{'State':^8} | {'LEFT':^10s} | {'STAY':^10s} | {'RIGHT':^10s}")
    print("-" * 50)
    
    for state in range(5):
        q_vals = [f"{Q_table.predict(state, a):+7.3f}" for a in range(3)]
        action_names = ["LEFT", "STAY", "RIGHT"]
        best_action = action_names[np.argmax([Q_table.predict(state, a) for a in range(3)])]
        print(f"{state:^8} | " + " | ".join(f"{q:^10s}" for q in q_vals) + f" <- {best_action}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FITTED Q ITERATION: Learning from Offline Data")
    print("="*70)
    
    env = GridWorld()
    
    # ========== 1. COLLECT DATASET ==========
    print("\n1. COLLECT DATASET")
    print("-" * 70)
    
    # Try two different collection strategies
    print("Strategy A: Pure random exploration")
    dataset_random = collect_dataset_random(env, num_samples=300)
    print(f"  Collected {len(dataset_random)} samples")
    
    print("\nStrategy B: Mixed (50% random + 50% go-right)")
    dataset_mixed = collect_dataset_mixed(env, num_samples=300)
    print(f"  Collected {len(dataset_mixed)} samples")
    
    # Show some samples
    print("\n  Sample transitions from dataset:")
    for i in range(3):
        s, a, r, s_next = dataset_mixed[i]
        action_name = ["LEFT", "STAY", "RIGHT"][a]
        print(f"    (s={s}, a={action_name:6s}, r={r:+3.0f}, s'={s_next})")
    
    # ========== 2. RUN FQI ON RANDOM DATASET ==========
    print("\n" + "="*70)
    print("2. FIT Q-FUNCTION ON RANDOM EXPLORATION DATASET")
    print("-" * 70)
    
    Q_table_random, losses_random = fitted_q_iteration_table(
        dataset_random,
        gamma=0.9,
        iterations=30,
        lr=0.2,
        verbose=True
    )
    
    policy_random = get_policy(Q_table_random)
    mean_rew_r, std_rew_r = test_policy(env, policy_random)
    
    print("\nLearned from random exploration:")
    show_q_table(Q_table_random)
    print(f"  Learned policy: {['LEFT', 'STAY', 'RIGHT'][policy_random[0]]}")
    print(f"  Test reward: {mean_rew_r:+.2f} ± {std_rew_r:.2f}")
    
    # ========== 3. RUN FQI ON MIXED DATASET ==========
    print("\n" + "="*70)
    print("3. FIT Q-FUNCTION ON MIXED DATASET")
    print("-" * 70)
    
    Q_table_mixed, losses_mixed = fitted_q_iteration_table(
        dataset_mixed,
        gamma=0.9,
        iterations=30,
        lr=0.2,
        verbose=True
    )
    
    policy_mixed = get_policy(Q_table_mixed)
    mean_rew_m, std_rew_m = test_policy(env, policy_mixed)
    
    print("\nLearned from mixed exploration:")
    show_q_table(Q_table_mixed)
    print(f"  Learned policy: {['LEFT', 'STAY', 'RIGHT'][policy_mixed[0]]}")
    print(f"  Test reward: {mean_rew_m:+.2f} ± {std_rew_m:.2f}")
    
    # ========== 4. COMPARE ==========
    print("\n" + "="*70)
    print("4. ANALYSIS: Effect of Dataset Quality")
    print("="*70)
    
    print(f"\nRandom dataset learning:")
    print(f"  Final loss: {losses_random[-1]:.6f}")
    print(f"  Policy reward: {mean_rew_r:+.2f}")
    print(f"  Result: {'✓ GOOD' if mean_rew_r > 5 else '✗ BAD'}")
    
    print(f"\nMixed dataset learning (better exploration):")
    print(f"  Final loss: {losses_mixed[-1]:.6f}")
    print(f"  Policy reward: {mean_rew_m:+.2f}")
    print(f"  Result: {'✓ GOOD' if mean_rew_m > 5 else '✗ Better with more data'}")
    
    # ========== 5. KEY INSIGHT ==========
    print("\n" + "="*70)
    print("KEY INSIGHTS: FQI vs Q-Learning")
    print("="*70)
    
    print("\nFitted Q Iteration (this example):")
    print("  ✓ Offline: Learn from pre-collected data")
    print("  ✓ Batch learning: Process entire dataset repeatedly")
    print("  ✓ Convergence depends on data coverage")
    print("  ✓ Used in: Robotics offline RL, learning from logs")
    
    print("\nQ-Learning (Example 4):")
    print("  ✓ Online: Learn while interacting")
    print("  ✓ Incremental: Update one sample at a time")
    print("  ✓ Guarantees convergence with exploration")
    print("  ✓ Used in: Real-time control, environment interaction")
    
    print("\nCritical observation:")
    print(f"  - Dataset quality matters! Mixed data gave reward: {mean_rew_m:+.2f}")
    print(f"  - Random exploration gave reward: {mean_rew_r:+.2f}")
    print(f"  - More coverage → Better learned policy")
    
    print("\n" + "="*70)
    print("FQI ALGORITHM SUMMARY")
    print("="*70)
    print("\n1. Collect dataset D = {(s_i, a_i, r_i, s'_i)}")
    print("2. Initialize Q(s,a) = 0 for all s,a")
    print("3. Repeat for N iterations:")
    print("   For each sample (s_i, a_i, r_i, s'_i) in D:")
    print("     y_i <- r_i + γ * max_a' Q(s'_i, a')")
    print("     Q(s_i, a_i) <- Q(s_i, a_i) + lr * (y_i - Q(s_i, a_i))")
    print("4. Extract policy: π(s) = argmax_a Q(s,a)")
    print("="*70)
