import gymnasium as gym
import numpy as np

import os
import matplotlib.pyplot as plt

alpha = 0.1            # learning rate
gamma = 0.99           # discount factor
epsilon_start = 1.0    # starting exploration rate
epsilon_decay = 0.995  # rate episilon decays after each episode
epsilon_min = 0.01     # minimum exploration rate
num_episodes = 5000    # number of training episodes

# alpha values for comparison
alpha_values = [0.1, 0.5, 0.9]

# gamma values for comparison
gamma_values = [0.50, 0.90, 0.99]

# epsilon decay values for comparison
epsilon_decay_values = [0.99, 0.995, 0.999]

# best hyperparameters
best_alpha = 0.5
best_gamma = 0.99
best_epsilon_decay = 0.99
eval_episodes = 30

class QLearningAgent:
    def __init__(self, n_states=500, n_actions=6, learning_rate=alpha, 
                 discount_factor=gamma, epsilon=epsilon_start, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # q table is a 2D array of shape (n_states, n_actions)
        self.q_table = np.zeros((n_states, n_actions))

    def get_action(self, state, training=True):
        # ie if training and [0.0, 1.0) < epsilon, return a random action 0-5
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        # otherwise pick the action with the highest Q-value
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state, action]

        # If the passenger was delivered (terminated) target Q is the reward b/c the episode is over
        # if truncated, the agent is still learning from the next state and needs to consider the max Q-value
        if terminated:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # q-learniing update formula
        # Q(s,a) <- Q(s,a) + alpha * [ reward + gamma * max_a' Q(s',a') - Q(s,a) ]
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate"""
        # ie max(0.01, <current_epsilon> * 0.995) so that epsilon never goes below 0.01
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(num_episodes=5000, learning_rate=alpha, discount_factor=gamma, epsilon_decay=epsilon_decay):
    env = gym.make("Taxi-v3")
    agent = QLearningAgent(learning_rate=learning_rate, discount_factor=discount_factor, epsilon_decay=epsilon_decay)
    episode_rewards = []

    for episode in range(num_episodes):
        # create random initial state for the taxi environment
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # select random action with probability epsilon, otherwise select the action with the highest Q-value
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # update the Q-table
            agent.update(state, action, reward, next_state, terminated)

            # Either one ends the episode loop
            done = terminated or truncated

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward (last 100): {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards

def plot_results(episode_rewards):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Episode reward")


    window = 50
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
    plt.plot(range(window - 1, len(episode_rewards)), moving_avg,
             label=f"{window}-episode moving average")

    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Discount factor gamma = {gamma}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/gamma99.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_alpha_comparison(results, window=100):
    os.makedirs("results", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
 
    inset = ax1.inset_axes([0.35, 0.12, 0.45, 0.45])
    start = num_episodes // 3
    for a, rewards in results.items():
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x = np.arange(window - 1, len(rewards))
        ax1.plot(x, moving_avg, label=f"α = {a}")
        inset.plot(x[x >= start], moving_avg[x >= start])
 
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Total reward ({window}-episode moving average)")
    ax1.set_title("learning curves")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    inset.set_title(f"episodes {start}+", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)
 
    fig.suptitle("Effect of alpha on Q-learning performance", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/alpha_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

 
def plot_gamma_comparison(results, window=100):
    os.makedirs("results", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    inset = ax1.inset_axes([0.35, 0.12, 0.45, 0.45])
    start = num_episodes // 3
    for g, rewards in results.items():
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x = np.arange(window - 1, len(rewards))
        ax1.plot(x, moving_avg, label=f"γ = {g}")
        inset.plot(x[x >= start], moving_avg[x >= start])

    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Total reward ({window}-episode moving average)")
    ax1.set_title("learning curves")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    inset.set_title(f"episodes {start}+", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)

    fig.suptitle("Effect of gamma on Q-learning performance", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/gamma_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

def plot_epsilon_comparison(results, window=100):
    os.makedirs("results", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
 
    inset = ax1.inset_axes([0.35, 0.12, 0.45, 0.45])
    start = num_episodes // 3
    for d, rewards in results.items():
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x = np.arange(window - 1, len(rewards))
        ax1.plot(x, moving_avg, label=f"ε decay = {d}")
        inset.plot(x[x >= start], moving_avg[x >= start])
 
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(f"Total reward ({window}-episode moving average)")
    ax1.set_title("learning curves")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    inset.set_title(f"episodes {start}+", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)
 
    fig.suptitle("Effect of epsilon decay on Q-learning performance", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/epsilon_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def evaluate_agent(agent, n_episodes=eval_episodes, label="Agent"):
    env = gym.make("Taxi-v3")
    rewards = []
    steps_list = []
    successes = 0
 
    for i in range(n_episodes):
        # seed=i gives every evaluated agent the same starting positions, so the comparison is fair
        state, _ = env.reset(seed=i)
        total_reward = 0
        steps = 0
        done = False
 
        while not done:
            # training=False -> always pick the action with the highest Q-value
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
 
        rewards.append(total_reward)
        steps_list.append(steps)
        if terminated:
            successes += 1
 
    env.close()
 
    print(f"\n{label} ({n_episodes} evaluation episodes, exploration off)")
    print(f"  Mean reward:     {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Success rate:    {100 * successes / n_episodes:.1f}%")
    print(f"  Mean steps:      {np.mean(steps_list):.2f}")
    return rewards, steps_list, successes
 
if __name__ == "__main__":

    # C) baseline
    print("\nTraining baseline...")
    baseline_agent, baseline_rewards = train_agent(num_episodes)
    print(f"\nalpha={alpha}, gamma={gamma}, epsilon decay={epsilon_decay}")
    print(f"Final avg reward (last 500 episodes): {np.mean(baseline_rewards[-500:]):.2f}")
    plot_results(baseline_rewards)

    # D) learning rate experiment
    alpha_results = {}
    for a in alpha_values:
        print(f"\nTraining w/ alpha = {a}")
        agent, rewards = train_agent(num_episodes, learning_rate=a)
        alpha_results[a] = rewards

    for a, rewards in alpha_results.items():
        print(f"\nalpha={a}")
        print(f"Final avg reward (last 500 episodes): {np.mean(rewards[-500:]):.2f}")
    plot_alpha_comparison(alpha_results)

    # E) discount factor experiment
    gamma_results = {}
    for g in gamma_values:
        print(f"\nTraining w/ gamma = {g}")
        agent, rewards = train_agent(num_episodes, discount_factor=g)
        gamma_results[g] = rewards

    for g, rewards in gamma_results.items():
        print(f"\ngamma={g}")
        print(f"Final avg reward (last 500 episodes): {np.mean(rewards[-500:]):.2f}")
    plot_gamma_comparison(gamma_results)

    # F) epsilon decay experiment
    epsilon_results = {}
    for d in epsilon_decay_values:
        print(f"\nTraining w/ epsilon decay = {d}")
        agent, rewards = train_agent(num_episodes, epsilon_decay=d)
        epsilon_results[d] = rewards

    for d, rewards in epsilon_results.items():
        print(f"\nepsilon decay={d}")
        print(f"Final avg reward (last 500 episodes): {np.mean(rewards[-500:]):.2f}")
    plot_epsilon_comparison(epsilon_results)

    # G) evaluate best agent vs baseline (reuses the baseline agent trained above)
    print(f"\nTraining best config (alpha={best_alpha}, gamma={best_gamma}, epsilon decay={best_epsilon_decay})...")
    best_agent, _ = train_agent(num_episodes, learning_rate=best_alpha,
                                discount_factor=best_gamma, epsilon_decay=best_epsilon_decay)

    evaluate_agent(baseline_agent, label="Baseline")
    evaluate_agent(best_agent, label="Best config")

    same = np.mean(np.argmax(baseline_agent.q_table, axis=1) == np.argmax(best_agent.q_table, axis=1))
    print(f"\nStates where both agents pick the same action: {100 * same:.1f}%")

    print("\nTraining complete!")
