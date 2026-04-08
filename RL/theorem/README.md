# Simple RL Examples Guide
## Learning Reinforcement Learning from Basics

This folder contains 10 simple examples to understand fundamental RL concepts:

---

## Quick Start

Run each example in order:

```bash
cd /home/quangvd7/self_learning/Robot_learning/RL

# 1. Basic concepts (State, Action, Reward, Policy)
python 01_basic_concepts.py

# 2. Value functions & Bellman equation
python 02_value_functions.py

# 3. Finding optimal policy
python 03_optimal_policy.py

# 4. Q-Learning (learning from interaction)
python 04_q_learning.py

# 5. Offline learning with Fitted Q Iteration
python 05_fitted_q_iteration_simple.py

# 6. Q-Learning vs Gradient Descent (understanding semi-gradient)
python 06_online_qlearning_vs_gradient_descent.py

# 7. DQN with epsilon-greedy (online)
python 07_deep_q_network_cartpole.py

# 8. DQN with replay buffer (greedy)
python 08_dqn_replay_buffer.py

# 9. DQN with epsilon-greedy + replay buffer (combined)
python 09_dqn_greedy_replay_buffer.py

# 10. Double DQN (DDQN)
python 10_double_dqn_cartpole.py
```

---

## Example 1: Basic Concepts (`01_basic_concepts.py`)

### What You Learn:
- **STATE** (s): Where the agent is
- **ACTION** (a): What the agent does
- **REWARD** (r): Feedback from environment
- **POLICY** (π): Rule for choosing actions

### The Example:
1D grid world: `0 ---- 1 ---- 2 ---- 3 ---- 4 (GOAL)`

```
Agent at position 0
Actions:  LEFT (-1), STAY (0), RIGHT (+1)
Reward:   +10 at goal, -1 per step
Policy:   Determines which action to take at each state
```

### Key Code:
```python
# Policy: a rule that maps state → action
policy = {0: "RIGHT", 1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "STAY"}

# Run one episode (trajectory)
trajectory, total_reward = run_episode(env, policy)
```

### Output:
```
Different policies lead to different results:
- Always Right: Average reward +2.5
- Always Stay: Average reward -8.0
```

**Key Insight:** Different policies produce different cumulative rewards!

---

## Example 2: Value Functions (`02_value_functions.py`)

### What You Learn:
- **V^π(s)**: Expected cumulative reward from state s, following policy π
- **Q^π(s,a)**: Expected cumulative reward from state s, taking action a, then following π
- **Bellman Equation**: Recursive relationship for computing values

### The Bellman Equation:
```
V(s) = R(s) + γ * V(s')

Where:
- R(s): Immediate reward
- γ (gamma): Discount factor (0.9)
- V(s'): Value of next state
```

### Key Code:
```python
# Bellman equation implementation
for state in env.states:
    action = policy[state]
    next_state, reward = env.get_transitions(state, action)
    V[state] = reward + gamma * V[next_state]
```

### Example Output:
```
Policy: {0:'RIGHT', 1:'RIGHT', 2:'RIGHT', 3:'RIGHT', 4:'STAY'}

State Values:
  V(0) = +8.19    (far from goal, but can reach it)
  V(1) = +8.10    
  V(2) = +9.00    (closer to goal)
  V(3) = +9.00    
  V(4) = +0.00    (already at goal)

Q-values for each action:
  State 0:
    Q(0, LEFT)  = -10.00  (hits wall penalty)
    Q(0, STAY)  = +7.19   (stay in place then follow policy)
    Q(0, RIGHT) = +8.10   (best action!)
```

**Key Insight:** V(s) tells how good a state is. Q(s,a) tells how good an action is!

---

## Example 3: Optimal Policy (`03_optimal_policy.py`)

### What You Learn:
- **Policy Improvement**: Choose best action using Q-values
- **Policy Iteration**: Alternately evaluate and improve until convergence
- **Value Iteration**: Direct computation of optimal values
- How to find the **BEST policy**

### The Idea:
```
If you know Q(s,a) for all state-action pairs,
then the optimal policy is:
    
    π*(s) = argmax_a Q(s,a)
            (pick the action with highest Q-value)
```

### Two Algorithms:

**Algorithm 1: Policy Iteration**
```
Initialize policy
Repeat:
  1. Evaluate: Compute V(s) for current policy
  2. Improve: π_new(s) = argmax_a Q(s,a) using V(s)
Until policy doesn't change
```

**Algorithm 2: Value Iteration**
```
Repeat:
  For each state:
    V(s) = max_a [R(s,a) + γ*V(s')]
Extract policy: π(s) = argmax_a [...]
Until convergence
```

### Example Output:
```
Initial policy (bad):  {0:'STAY', 1:'STAY', 2:'STAY', 3:'STAY', 4:'STAY'}
V(0) = 0.00  (stuck in place, no progress)

After improvement:
New policy: {0:'RIGHT', 1:'RIGHT', 2:'RIGHT', 3:'RIGHT', 4:'STAY'}
V(0) = 8.19  (much better!)

✓ Both Policy Iteration and Value Iteration converge to same optimal policy!
```

**Key Insight:** We can find the optimal policy systematically!

---

## Example 4: Q-Learning (`04_q_learning.py`)

### What You Learn:
- **Model-free learning**: Learn without knowing environment dynamics
- **Temporal Difference (TD) learning**: Update estimates using experience
- **Q-Learning algorithm**: Learn Q-values through trial-and-error
- **Exploration vs Exploitation**: ε-greedy strategy

### The Problem with Previous Examples:
- We assumed we KNOW state transitions and rewards
- Real world: You don't know the dynamics!

### Q-Learning Solution:
```
Agent learns Q(s,a) by:
1. Taking random actions
2. Observing transitions and rewards
3. Updating Q-values based on experience

Q(s,a) ← Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]
          └─old─┘   └─────────target──────────┘
                     TD-error: how wrong we were
```

### Key Concepts:

**ε-greedy exploration:**
- With probability ε: Take random action (explore)
- With probability 1-ε: Take best known action (exploit)

**Learning rate α:**
- How much to update each time
- α=1: Replace old estimate completely
- α=0.1: Gradually blend new information

### Example Output:
```
Before training:
  Q-table: all zeros
  Random policy reward: -4.2

After 200 episodes of training:
  Q(0, LEFT)  = -8.25  (bad action)
  Q(0, STAY)  = +5.21  (okay)
  Q(0, RIGHT) = +8.03  (best action!)

Learned policy: {0:'RIGHT', 1:'RIGHT', 2:'RIGHT', 3:'RIGHT', 4:'STAY'}
  Learned policy reward: +8.1
  Success rate: 100%

Learning curves show:
- Reward per episode increases with training
- Steps to complete decrease (more efficient)
```

**Key Insight:** Agent learns from experience without needing to know the model!

---

## Example 5: Offline Learning with Fitted Q Iteration (`05_fitted_q_iteration_simple.py`)

### What You Learn:
- **Offline learning**: Learn from a fixed dataset (no interaction)
- **Batch Q-learning**: Process multiple samples repeatedly
- **Fitted Q Iteration**: Offline version of Q-Learning
- **Importance of data quality**: Better exploration → Better policy

### The Problem Solved:
- Previous examples: Learn through trial-and-error interaction
- Real world: Sometimes you only have logged data (no simulator)
- Example: Imitation learning, offline RL

### Fitted Q Iteration Algorithm:
```python
# Offline learning from dataset D = {(s,a,r,s'), ...}

for iteration in range(max_iterations):
    for (state, action, reward, next_state) in Dataset:
        # Compute target (same as Q-Learning)
        target = reward + gamma * max_a' Q(next_state, a')
        
        # Update Q-value
        Q(state, action) ← Q(state, action) + lr * (target - Q(state, action))
```

### Key Difference from Q-Learning:
```
Q-Learning (Online):          Fitted Q Iteration (Offline):
  Each step → single update    Process entire dataset repeatedly
  One (s,a,r,s') at a time    All transitions in batches
  Learn as you explore        Learn from logged experience
```

### Example Output:
```
Two dataset collection strategies tested:

1. Random Exploration:
   - Takes random actions to collect data
   - Less structured, but diverse coverage
   - Final reward: +8.56
   - Q-values well distributed

2. Mixed Exploration (50% random + 50% go-right):
   - More focused on good behavior
   - Better initial data quality
   - Final reward: +8.74
   - Slightly better convergence

Key insight: Dataset quality affects learned policy!
```

**Key Insight:** You can learn effective policies from logged data alone, without exploration!

---

## Example 6: Online Q-Learning vs Gradient Descent (`06_online_qlearning_vs_gradient_descent.py`)

### What You Learn:
- **Why Q-Learning is NOT standard Gradient Descent**
- **Semi-Gradient principle**: Fixed targets vs moving targets
- **Stability in RL optimization**
- **Connection to Deep Q-Networks (DQN)**

### The Key Question (From Your Lecture):
```
Q-Learning update:
  φ ← φ - η(Q_φ(s,a) - [r + γ·max Q_φ(s',a')]) ∇_φ Q_φ(s,a)

Why is this NOT gradient descent?
  → Because the target y = r + γ·max Q_φ(s',a') is FIXED
     (we don't differentiate through it!)
```

### Two Approaches Compared:

**Semi-Gradient Q-Learning (Correct):**
```
Target: y = r + γ·max Q(s',a')  ← TREATED AS FIXED
Gradient: Only flows through Q(s,a) in numerator
Update: Q(s,a) ← Q(s,a) - α · (Q(s,a) - y)

Benefit: Stable, predictable convergence
```

**Full Gradient Descent (Problematic):**
```
Target: y = r + γ·max Q(s',a')  ← ALSO depends on φ!
Gradient: Flows through both Q and target
Update: More aggressive, conflicting signals

Problem: Target moves as we optimize
         Creates instability in learning
```

### Why This Matters:

**In Theory:**
- Q-Learning convergence theorem assumes semi-gradient
- Full gradient breaks theoretical guarantees
- Semi-gradient is mathematically justified

**In Practice:**
- DQN uses "target networks" to implement semi-gradient:
  ```
  Main network:   Q_φ(s,a) - updated every step
  Target network: Q_φ_old(s,a) - updated every N steps
  ```
- This explicit separation creates the fixed targets!

### Example Output:
```
Both methods on simple 1D grid (similar performance):
  Semi-Gradient Q-Learning: 8.80 average reward
  Full Gradient Descent:     8.86 average reward

But on complex problems:
  Semi-Gradient: Stable, smooth convergence
  Full Gradient: Diverges or oscillates

Generated: qlearning_vs_gradient_descent.png
```

**Key Insight:** Understanding semi-gradient principle is essential for deep RL!

---

## Example 7: DQN with Epsilon-Greedy (`07_deep_q_network_cartpole.py`)

### What You Learn:
- Epsilon-greedy exploration with a neural Q-function
- End-to-end training with plots and optional GIFs
- Custom CartPole environment (no gym dependency)

### Key Formula (epsilon-greedy action selection):
$$
a=\begin{cases}
	ext{random action}, & \text{with prob }\epsilon \\
\arg\max_a Q_\phi(s,a), & \text{with prob }1-\epsilon
\end{cases}
$$
Explanation: with probability $\epsilon$ the agent explores; otherwise it exploits by choosing the greedy action under $Q_\phi$.

### TD Target (online update):
$$
y = r + \gamma \max_{a'} Q_{\phi^-}(s', a')
$$
Explanation: $\phi^-$ is a slowly updated target network so the bootstrapped target changes more smoothly.

---

## Example 8: DQN with Replay Buffer (`08_dqn_replay_buffer.py`)

### What You Learn:
- A lean DQN loop focused on experience replay
- Greedy policy to show replay alone is not enough for exploration

### Key Formula (replay buffer training):
$$
(s,a,r,s') \sim \mathcal{D}
$$
$$
L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\big[(Q_\phi(s,a)-y)^2\big]
$$
Explanation: the replay buffer $\mathcal{D}$ stores past transitions; sampling breaks correlation so SGD behaves more like supervised learning. The target is $y = r + \gamma \max_{a'} Q_{\phi^-}(s',a')$.

---

## Example 9: Epsilon-Greedy + Replay Buffer (`09_dqn_greedy_replay_buffer.py`)

### What You Learn:
- How epsilon-greedy exploration fills the replay buffer
- Why combining exploration + replay stabilizes DQN training

### Combined Formula (data collection + training):
$$
L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\big[(Q_\phi(s,a) - (r + \gamma \max_{a'} Q_{\phi^-}(s',a')))^2\big]
$$
Explanation: transitions in $\mathcal{D}$ are collected with epsilon-greedy actions (Example 7), and the loss uses replay sampling (Example 8).

---

## Example 10: Double DQN (DDQN) (`10_double_dqn_cartpole.py`)

### What You Learn:
- Reduce overestimation by decoupling action selection and evaluation
- Use online network for $\arg\max$ and target network for value

### Standard DQN Target:
$$
y_{\text{DQN}} = r + \gamma \max_{a'} Q_{\phi^-}(s', a')
$$
Explanation: the same network both selects and evaluates the next action, which can overestimate values.

### Double DQN Target:
$$
y_{\text{DDQN}} = r + \gamma Q_{\phi^-}(s', \arg\max_{a'} Q_{\phi}(s', a'))
$$
Explanation: the online network $Q_{\phi}$ selects the action, and the target network $Q_{\phi^-}$ evaluates it, reducing maximization bias.

---

## Concept Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING PROGRESSION                    │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│   EXAMPLE 1: BASICS              │
│   States, Actions, Rewards       │
│   Policies                        │
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  EXAMPLE 2: VALUE FUNCTIONS      │
│  V(s), Q(s,a)                    │
│  Bellman Equation                │
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  EXAMPLE 3: OPTIMAL POLICY       │
│  Policy Iteration                │
│  Value Iteration                 │
│  argmax Q(s,a)                   │
└──────────────────────────────────┘
         │
    ┌────┴─────────────────┐
    │                      │
    ▼                      ▼
┌─────────────────┐   ┌──────────────────────┐
│ EXAMPLE 4:      │   │ EXAMPLE 5:           │
│ ONLINE          │   │ OFFLINE              │
│ Q-LEARNING      │   │ FITTED Q ITERATION   │
│                 │   │                      │
│ Model-free      │   │ Batch learning       │
│ Interactive     │   │ From logs/dataset    │
│ Learning        │   │ No interaction       │
└─────────────────┘   └──────────────────────┘
    │                      │
    └────────┬─────────────┘
             ▼
    ┌──────────────────────────────┐
    │  EXAMPLE 6:                  │
    │  QLEARNING vs GRADIENT DESC   │
    │                              │
    │  Semi-Gradient Principle     │
    │  Fixed Targets (Foundation)  │
    │  Connection to DQN           │
    └──────────────────────────────┘
```

---

## Mathematical Summary

### Core Equations:

**Bellman Expectation Equation (Policy Evaluation):**
```
V^π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]

Simplified (deterministic):
V^π(s) = r(s,π(s)) + γ*V^π(s')
```

**Q-function:**
```
Q^π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γV^π(s')]

Simplified (deterministic):
Q^π(s,a) = r(s,a) + γ*V^π(next_state)
```

**Policy Improvement:**
```
π'(s) = argmax_a Q^π(s,a)
```

**Bellman Optimality Equation:**
```
V*(s) = max_a [r(s,a) + γ*V*(s')]
```

**Q-Learning Update:**
```
Q(s,a) ← Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]
                        └────────────────────────┘
                         Bellman Optimality target
```

---

## Common Mistakes & How to Fix Them

### Mistake 1: Confusing V(s) and Q(s,a)
```
❌ V(s) = value of being in state s
✓ V(s) = expected reward following policy π
✓ Q(s,a) = expected reward if you take action a first, then follow π
```

### Mistake 2: Using wrong Bellman update
```
❌ Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ*Q(s',π(s')) - Q(s,a)]
✓ Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ*max_a' Q(s',a') - Q(s,a)]
   Difference: Use MAX over all actions, not your policy's action
```

### Mistake 3: Not exploring enough
```
❌ Always take greedy action: Takes too long to learn
✓ Use ε-greedy: Occasionally take random actions
   This balances exploration (finding good actions) vs exploitation (using them)
```

### Mistake 4: Wrong discount factor γ
```
❌ γ = 0: Only care about immediate reward (shortsighted)
✓ γ = 0.9-0.99: Value long-term rewards (reasonable)
❌ γ = 1.0: Infinite future values (numerically unstable)
```

---

## Extensions (Try These!)

1. **Different Environment Sizes**: Modify `num_states` in examples
2. **Different Rewards**: Change reward values (e.g., +100 for goal instead of +10)
3. **Multi-agent**: Agent must avoid obstacles
4. **Stochastic Environment**: Actions don't always succeed
5. **Continuous State Space**: Use function approximation instead of Q-table

---

## References

Lecture slide concepts covered:
- ✓ Policy (π)
- ✓ Value functions V(s), Q(s,a)
- ✓ Bellman equations (recursive relationships)
- ✓ Policy evaluation (dynamic programming)
- ✓ Policy improvement (greedy policy)
- ✓ Model-free learning (Q-Learning)
- ✓ Model-free, Offline learning (Fitted Q Iteration)
- ✓ Semi-Gradient Methods (Q-Learning vs Gradient Descent)

### Key Papers & Resources:
- **Sutton & Barto (2018)**: "Reinforcement Learning: An Introduction"
  - Chapter 6: Temporal Difference Learning
  - Chapter 11: Semi-Gradient Methods
- **Mnih et al. (2015)**: "Human-level control through deep RL" (DQN paper)
  - Introduces target networks (practical semi-gradient implementation)

---

**Good luck learning RL! Start with Example 1, understand it deeply, then move to the next.**
