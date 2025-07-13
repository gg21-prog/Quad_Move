from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym

env = gym.make("Blackjack-v1", sab=True)
#same env as book
done = False
observation, info = env.reset() #obs = {current sum, dealer's card, ace (y/n)}

action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)

class BlackjackAgent:
    def __init__(
        self,
        env,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1, #given in book
    ):
        """Initializing a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values) and an epsilon.

        Args:
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns_count = defaultdict(lambda: np.zeros(env.action_space.n))

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            episode: list[tuple[tuple[int, int, bool], int, float]]
    ):
        G = 0
        visited = set()
        for obs, action, reward in reversed(episode):
            G = self.discount_factor * G + reward
            if (obs, action) not in visited: #first visit only
                visited.add((obs, action))
                self.returns_sum[obs][action] += G
                self.returns_count[obs][action] += 1
                prev_q = self.q_values[obs][action] #before update
                self.q_values[obs][action] = (
                    self.returns_sum[obs][action] / self.returns_count[obs][action]
                )
                self.training_error.append(abs(self.q_values[obs][action] - prev_q))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

n_episodes = 500_000 #given in book
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes) # was initially n_episodes/2 but this gives better win rate 
final_epsilon = 0.1 #reducing this did not improve win rate

agent = BlackjackAgent(
    env=env,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    epi = []

    # one episode at a time
    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        epi.append((obs, action, reward))
        done = terminated or truncated
        obs = next_obs

    agent.update(epi)
    agent.decay_epsilon()

#entirely taken visualization from gymnasium documentation

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()

def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        #lambda obs: state_value.get((obs[0], obs[1], usable_ace), 0)
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        #lambda obs: policy.get((obs[0], obs[1], usable_ace), 0)
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig



# state values & policy with usable ace (ace counts as 11)
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
plt.show()

#for win rate

eval_n = 10_000
wins, draws, losses = 0,0,0
total_reward = 0

agent.epsilon = 0.0

for _ in tqdm(range(eval_n), desc="Evaluating Policy"):
    obs, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent.get_action(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    total_reward += episode_reward
    if episode_reward > 0:
        wins += 1
    elif episode_reward == 0:
        draws += 1
    else:
        losses += 1

win_rate = wins / eval_n * 100
draw_rate = draws / eval_n * 100
loss_rate = losses / eval_n * 100
average_reward = total_reward / eval_n

print(win_rate)