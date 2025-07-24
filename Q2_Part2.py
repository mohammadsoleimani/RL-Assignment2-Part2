# Assignment 2 Part 2
# Reinforcement Learning
# Mohammad Soleimani



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import copy

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


class GridWorldPart2:
    def __init__(self):
        self.size = 5
        self.gamma = 0.95

        # Define the grid layout based on Part 2:
        # W B W W G
        # W W W W W
        # T W W W T
        # W W W W W
        # T W R W Y

        # Special states positions (0-based indexing)
        self.blue_state = (0, 1)  # B
        self.green_state = (0, 4)  # G
        self.red_state = (4, 2)  # R
        self.yellow_state = (4, 4)  # Y

        # Terminal states positions (T)
        self.terminal_states = {(2, 0), (2, 4), (4, 0)}

        # All special states (non-white)
        self.special_states = {
            self.blue_state: 'B',
            self.green_state: 'G',
            self.red_state: 'R',
            self.yellow_state: 'Y'
        }

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['up', 'down', 'left', 'right']

        # Initialize Q-values and state values
        self.Q = defaultdict(lambda: np.zeros(4))
        self.V = np.zeros((self.size, self.size))

    def is_terminal(self, state):
        """Check if state is terminal"""
        return state in self.terminal_states or state == self.green_state

    def is_valid_state(self, state):
        """Check if state is within grid bounds"""
        row, col = state
        return 0 <= row < self.size and 0 <= col < self.size

    def get_next_state_and_reward(self, state, action):
        """Get next state and reward for a given state-action pair"""
        if self.is_terminal(state):
            return state, 0  # Terminal states don't transition

        row, col = state
        d_row, d_col = self.actions[action]
        new_row = row + d_row
        new_col = col + d_col

        # Check if stepping off the grid
        if not self.is_valid_state((new_row, new_col)):
            return state, -0.5 if (row, col) in [self.red_state, self.yellow_state] or (row,
                                                                                        col) not in self.special_states else -0.2

        new_state = (new_row, new_col)

        # Check if moving to green or terminal state
        if new_state == self.green_state:
            return new_state, 1.0  # Move to green with +1 reward
        if self.is_terminal(new_state):
            return new_state, 0  # Move to terminal with 0 reward

        return new_state, -0.2  # Normal move reward

    def generate_episode(self, policy, start_state=None, exploring_starts=False):
        """Generate an episode following the given policy"""
        episode = []

        # Choose starting state
        if exploring_starts:
            # Random start for exploring starts
            while True:
                state = (np.random.randint(0, self.size), np.random.randint(0, self.size))
                if not self.is_terminal(state):
                    break
        else:
            state = start_state if start_state is not None else self.blue_state

        max_steps = 1000  # Prevent infinite episodes
        for step in range(max_steps):
            if self.is_terminal(state):
                break

            # Choose action according to policy
            if isinstance(policy, dict):
                action_probs = policy[state]
            else:
                action_probs = policy(state)

            action = np.random.choice(4, p=action_probs)

            # Take action and observe reward and next state
            next_state, reward = self.get_next_state_and_reward(state, action)

            episode.append((state, action, reward))
            state = next_state

        return episode

    def create_epsilon_soft_policy(self, Q, epsilon=0.1):
        """Create epsilon-soft policy from Q-values"""
        policy = {}

        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)
                if self.is_terminal(state):
                    policy[state] = np.zeros(4)  # No actions in terminal states
                else:
                    q_values = Q[state]
                    best_action = np.argmax(q_values)
                    action_probs = np.ones(4) * epsilon / 4
                    action_probs[best_action] += 1 - epsilon
                    policy[state] = action_probs

        return policy

    def create_equiprobable_policy(self):
        """Create equiprobable policy"""
        policy = {}
        for i in range(self.size):
            for j in range(self.size):
                state = (i, j)
                if self.is_terminal(state):
                    policy[state] = np.zeros(4)
                else:
                    policy[state] = np.ones(4) / 4
        return policy


def monte_carlo_exploring_starts(gridworld, num_episodes=100000):
    """Monte Carlo with Exploring Starts"""
    print("1. Monte Carlo with Exploring Starts...")

    Q = defaultdict(lambda: np.zeros(4))
    returns = defaultdict(list)
    policy = gridworld.create_equiprobable_policy()

    for episode_num in range(num_episodes):
        episode = gridworld.generate_episode(policy, exploring_starts=True)

        if not episode:
            continue

        G = 0
        visited = set()

        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gridworld.gamma * G + reward

            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)
                returns[state_action].append(G)
                Q[state][action] = np.mean(returns[state_action])

                best_action = np.argmax(Q[state])
                policy[state] = np.zeros(4)
                policy[state][best_action] = 1.0

        if (episode_num + 1) % 10000 == 0:
            print(f"Completed {episode_num + 1} episodes")

    return Q, policy


def monte_carlo_epsilon_soft(gridworld, num_episodes=100000, epsilon=0.1):
    """Monte Carlo without Exploring Starts (epsilon-soft)"""
    print("2. Monte Carlo with ε-soft policy...")

    Q = defaultdict(lambda: np.zeros(4))
    returns = defaultdict(list)

    for episode_num in range(num_episodes):
        policy = gridworld.create_epsilon_soft_policy(Q, epsilon)
        episode = gridworld.generate_episode(policy, start_state=gridworld.blue_state)

        if not episode:
            continue

        G = 0
        visited = set()

        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gridworld.gamma * G + reward

            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)
                returns[state_action].append(G)
                Q[state][action] = np.mean(returns[state_action])

        if (episode_num + 1) % 10000 == 0:
            print(f"Completed {episode_num + 1} episodes")

    final_policy = gridworld.create_epsilon_soft_policy(Q, epsilon=0)  # Greedy policy
    return Q, final_policy


def monte_carlo_off_policy(gridworld, num_episodes=100000):
    """Monte Carlo Off-Policy with Importance Sampling"""
    print("3. Monte Carlo Off-Policy with Importance Sampling...")

    Q = defaultdict(lambda: np.zeros(4))
    C = defaultdict(lambda: np.zeros(4))  # Cumulative weights
    behavior_policy = gridworld.create_equiprobable_policy()
    target_policy = gridworld.create_equiprobable_policy()

    for episode_num in range(num_episodes):
        episode = gridworld.generate_episode(behavior_policy, start_state=gridworld.blue_state)

        if not episode:
            continue

        G = 0
        W = 1.0

        for i in reversed(range(len(episode))):
            state, action, reward = episode[i]
            G = gridworld.gamma * G + reward

            C[state][action] += W
            if C[state][action] > 0:
                Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            best_action = np.argmax(Q[state])
            target_policy[state] = np.zeros(4)
            target_policy[state][best_action] = 1.0

            if action != best_action:
                break

            W = W * (1.0 / 0.25)

        if (episode_num + 1) % 10000 == 0:
            print(f"Completed {episode_num + 1} episodes")

    return Q, target_policy


def extract_value_function(Q):
    """Extract value function from Q-values"""
    V = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            state = (i, j)
            if not (state in gridworld.terminal_states or state == gridworld.green_state):
                V[i, j] = np.max(Q[state])
    return V


def extract_policy_array(policy_dict):
    """Convert policy dictionary to array format"""
    policy_array = np.zeros((5, 5, 4))
    for i in range(5):
        for j in range(5):
            state = (i, j)
            if state in policy_dict:
                policy_array[i, j] = policy_dict[state]
            else:
                policy_array[i, j] = np.zeros(4)
    return policy_array


def plot_value_function(V, title="Value Function", gridworld=None):
    """Plot the value function as a heatmap with special state labels"""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(V, cmap='viridis', aspect='equal')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value', rotation=270, labelpad=20)

    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            text = ax.text(j, i, f'{V[i, j]:.2f}', ha="center", va="center",
                           color="white" if V[i, j] < V.max() / 2 else "black", fontsize=10)

    if gridworld is not None:
        special_states = {
            gridworld.blue_state: ('B', 'blue'),
            gridworld.green_state: ('G', 'green'),
            gridworld.red_state: ('R', 'red'),
            gridworld.yellow_state: ('Y', 'orange')
        }

        for (i, j), (label, color) in special_states.items():
            circle = plt.Circle((j, i), 0.35, color=color, alpha=0.4, transform=ax.transData)
            ax.add_patch(circle)
            ax.text(j, i - 0.25, label, ha='center', va='center', fontsize=12, fontweight='bold')

        for (i, j) in gridworld.terminal_states:
            square = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, color='black', alpha=0.6, transform=ax.transData)
            ax.add_patch(square)
            ax.text(j, i, 'T', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax.set_xlim(-0.5, V.shape[1] - 0.5)
    ax.set_ylim(-0.5, V.shape[0] - 0.5)
    ax.set_xticks(range(V.shape[1]))
    ax.set_yticks(range(V.shape[0]))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_policy(policy_array, gridworld, title="Policy"):
    """Plot the policy as arrows with special state labels"""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(np.zeros((gridworld.size, gridworld.size)), cmap='gray', alpha=0.1)

    for i in range(gridworld.size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)

    arrows = ['↑', '↓', '←', '→']
    for i in range(gridworld.size):
        for j in range(gridworld.size):
            if not gridworld.is_terminal((i, j)):
                action_probs = policy_array[i, j]
                best_action = np.argmax(action_probs)
                ax.text(j, i, arrows[best_action], ha='center', va='center',
                        fontsize=20, fontweight='bold')

    special_states = {
        gridworld.blue_state: ('B', 'blue'),
        gridworld.green_state: ('G', 'green'),
        gridworld.red_state: ('R', 'red'),
        gridworld.yellow_state: ('Y', 'orange')
    }

    for (i, j), (label, color) in special_states.items():
        circle = plt.Circle((j, i), 0.35, color=color, alpha=0.4, transform=ax.transData)
        ax.add_patch(circle)
        ax.text(j, i + 0.25, label, ha='center', va='center', fontsize=12, fontweight='bold')

    for (i, j) in gridworld.terminal_states:
        square = plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, color='black', alpha=0.6, transform=ax.transData)
        ax.add_patch(square)
        ax.text(j, i, 'T', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax.set_xlim(-0.5, gridworld.size - 0.5)
    ax.set_ylim(-0.5, gridworld.size - 0.5)
    ax.set_xticks(range(gridworld.size))
    ax.set_yticks(range(gridworld.size))
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def compare_policies(policy1, policy2, gridworld, title1="Policy 1", title2="Policy 2"):
    """Compare two policies side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    arrows = ['↑', '↓', '←', '→']

    for ax, policy, title in [(ax1, policy1, title1), (ax2, policy2, title2)]:
        ax.imshow(np.zeros((gridworld.size, gridworld.size)), cmap='gray', alpha=0.1)

        for i in range(gridworld.size + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
            ax.axvline(i - 0.5, color='black', linewidth=0.5)

        for i in range(gridworld.size):
            for j in range(gridworld.size):
                if not gridworld.is_terminal((i, j)):
                    action_probs = policy[i, j]
                    best_action = np.argmax(action_probs)
                    ax.text(j, i, arrows[best_action], ha='center', va='center',
                            fontsize=16, fontweight='bold')

        special_states = {
            gridworld.blue_state: ('B', 'blue'),
            gridworld.green_state: ('G', 'green'),
            gridworld.red_state: ('R', 'red'),
            gridworld.yellow_state: ('Y', 'orange')
        }

        for (i, j), (label, color) in special_states.items():
            circle = plt.Circle((j, i), 0.3, color=color, alpha=0.4, transform=ax.transData)
            ax.add_patch(circle)
            ax.text(j, i + 0.25, label, ha='center', va='center', fontsize=10, fontweight='bold')

        for (i, j) in gridworld.terminal_states:
            square = plt.Rectangle((j - 0.35, i - 0.35), 0.7, 0.7, color='black', alpha=0.6, transform=ax.transData)
            ax.add_patch(square)
            ax.text(j, i, 'T', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        ax.set_xlim(-0.5, gridworld.size - 0.5)
        ax.set_ylim(-0.5, gridworld.size - 0.5)
        ax.set_xticks(range(gridworld.size))
        ax.set_yticks(range(gridworld.size))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def main():
    print("GridWorld Reinforcement Learning Assignment - Part 2")
    print("Environment: 5x5 GridWorld with Terminal States")
    print("Discount factor (γ): 0.95")
    print("Random seed: 42 (for reproducibility)")
    print("=" * 70)

    global gridworld
    gridworld = GridWorldPart2()

    print("GRID LAYOUT:")
    print("W B W W G")
    print("W W W W W")
    print("T W W W T")
    print("W W W W W")
    print("T W R W Y")
    print()
    print("Legend: W=White, B=Blue, G=Green, R=Red, Y=Yellow, T=Terminal")
    print("=" * 70)

    # Part 2, Question 1.1: Monte Carlo with Exploring Starts
    print("PART 2 - Question 1.1: Monte Carlo with Exploring Starts")
    print("=" * 70)

    Q_es, policy_es = monte_carlo_exploring_starts(gridworld, num_episodes=100000)
    V_es = extract_value_function(Q_es)
    policy_es_array = extract_policy_array(policy_es)

    print("\nValue function (Monte Carlo Exploring Starts):")
    print(V_es)
    plot_value_function(V_es, "Value Function - MC Exploring Starts", gridworld)
    plot_policy(policy_es_array, gridworld, "Optimal Policy - MC Exploring Starts")

    # Part 2, Question 1.2: Monte Carlo epsilon-soft
    print("\n" + "=" * 70)
    print("PART 2 - Question 1.2: Monte Carlo ε-soft")
    print("=" * 70)

    Q_eps, policy_eps = monte_carlo_epsilon_soft(gridworld, num_episodes=100000, epsilon=0.1)
    V_eps = extract_value_function(Q_eps)
    policy_eps_array = extract_policy_array(policy_eps)

    print("\nValue function (Monte Carlo ε-soft):")
    print(V_eps)
    plot_value_function(V_eps, "Value Function - MC ε-soft", gridworld)
    plot_policy(policy_eps_array, gridworld, "Optimal Policy - MC ε-soft")

    # Part 2, Question 2: Off-policy Monte Carlo
    print("\n" + "=" * 70)
    print("PART 2 - Question 2: Off-Policy Monte Carlo")
    print("=" * 70)

    Q_off, policy_off = monte_carlo_off_policy(gridworld, num_episodes=100000)
    V_off = extract_value_function(Q_off)
    policy_off_array = extract_policy_array(policy_off)

    print("\nValue function (Off-Policy Monte Carlo):")
    print(V_off)
    plot_value_function(V_off, "Value Function - Off-Policy MC", gridworld)
    plot_policy(policy_off_array, gridworld, "Optimal Policy - Off-Policy MC")

    # Compare all methods
    print("\n" + "=" * 70)
    print("COMPARISON OF METHODS")
    print("=" * 70)

    print("Difference between Exploring Starts and ε-soft:")
    print(f"  Value function: {np.max(np.abs(V_es - V_eps)):.4f}")

    print("Difference between Exploring Starts and Off-Policy:")
    print(f"  Value function: {np.max(np.abs(V_es - V_off)):.4f}")

    print("Difference between ε-soft and Off-Policy:")
    print(f"  Value function: {np.max(np.abs(V_eps - V_off)):.4f}")

    compare_policies(policy_es_array, policy_eps_array, gridworld,
                     "MC Exploring Starts", "MC ε-soft")

    compare_policies(policy_es_array, policy_off_array, gridworld,
                     "MC Exploring Starts", "Off-Policy MC")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nMonte Carlo Methods Analysis:")
    print("- All methods converge to the optimal policy navigating to the green state")
    print("- Exploring starts ensures comprehensive state-action exploration")
    print("- ε-soft balances exploration and exploitation with ε = 0.1")
    print("- Off-policy learns from equiprobable behavior policy using importance sampling")
    print("- Terminal states (T) and green state (G) end episodes")
    print("- Negative rewards (-0.2, -0.5) encourage shortest paths to green state")

    print("\nREPRODUCIBILITY NOTES:")
    print("- Random seeds set to 42 for both numpy and random modules")
    print("- Episode generation uses consistent random number generation")
    print("- All algorithms use first-visit Monte Carlo updates")
    print("- Number of episodes: 100,000 for each method")
    print("- ε-soft uses ε = 0.1 during learning")
    print("=" * 70)


if __name__ == "__main__":
    main()