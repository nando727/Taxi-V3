# Q-Learning on Gymnasium Taxi-v3

CPSC 4420 – Assignment 2
Fernando Tello

A tabular Q-learning agent that learns to pick up and drop off passengers in the Gymnasium **Taxi-v3** environment, along with experiments on how the learning rate (α), discount factor (γ), and epsilon decay affect learning.

## Environment

Taxi-v3 is a 5×5 grid with 4 designated pickup/dropoff locations. The taxi has to drive to the passenger, pick them up, drive to the destination, and drop them off.

- **States:** 500 discrete states (25 taxi positions × 5 passenger locations × 4 destinations)
- **Actions:** 6 discrete actions (south, north, east, west, pickup, dropoff)
- **Rewards:** −1 per step, +20 for a successful dropoff, −10 for an illegal pickup or dropoff
- **Episode ends:** when the passenger is delivered (terminated) or after 200 steps (truncated)

## Requirements

- Python 3.9+
- gymnasium (with the toy-text extras for rendering)
- numpy
- matplotlib

## Setup

Clone the repository:

```bash
git clone https://github.com/nando727/Taxi-V3.git
cd <your-repo-folder>
```

Create and activate an environment (conda shown, but a regular venv works too):

```bash
conda create -n cpsc4420 python=3.11
conda activate cpsc4420
```

Install the dependencies:

```bash
pip install "gymnasium[toy-text]" numpy matplotlib
```

## How to Run

```bash
python taxi.py
```

The script runs every experiment in order:

1. **Baseline** – trains one agent with the default settings and plots reward per episode
2. **Learning-rate experiment** – α = 0.1, 0.5, 0.9
3. **Discount-factor experiment** – γ = 0.50, 0.90, 0.99
4. **Epsilon-decay experiment** – ε decay = 0.99, 0.995, 0.999
5. **Best-agent evaluation** – trains the best configuration and evaluates it against the baseline over 30 episodes with exploration turned off

Each plot opens in a window, and the script continues to the next experiment once you close it. All plots are also saved to the `results/` folder. A full run takes a few minutes.

## Changing Hyperparameters

All settings are variables at the top of `quiz2.py`:

| Variable | Default | Description |
|---|---|---|
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Starting exploration rate |
| `epsilon_decay` | 0.995 | Epsilon is multiplied by this after each episode |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `num_episodes` | 5000 | Training episodes per run |
| `alpha_values` | [0.1, 0.5, 0.9] | Values tested in the learning-rate experiment |
| `gamma_values` | [0.50, 0.90, 0.99] | Values tested in the discount-factor experiment |
| `epsilon_decay_values` | [0.99, 0.995, 0.999] | Values tested in the epsilon-decay experiment |
| `best_alpha`, `best_gamma`, `best_epsilon_decay` | 0.5, 0.99, 0.99 | Configuration used for the best agent |
| `eval_episodes` | 30 | Evaluation episodes (exploration off) |

## Code Overview

- `QLearningAgent` – holds the 500×6 Q-table (initialized to zeros), ε-greedy action selection, the Q-learning update, and epsilon decay
- `train_agent()` – trains an agent; takes α, γ, and ε decay as arguments so experiments can loop over values
- `evaluate_agent()` – runs the trained policy with exploration off and no Q-table updates, and reports mean reward, success rate, and mean steps
- `plot_results()` – baseline reward-per-episode plot
- `plot_alpha_comparison()`, `plot_gamma_comparison()`, `plot_epsilon_comparison()` – comparison plots for each experiment

The Q-learning update used is:

```
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]
```

When an episode is terminated (passenger delivered), the target is just the reward `r`. When it is truncated at 200 steps, the target still includes `γ max_a' Q(s', a')`, since the taxi did not reach a real terminal state.

## Results

| File | Description |
|---|---|
| `results/baseline.png` | Baseline reward per episode |
| `results/alpha_comparison.png` | Learning-rate comparison |
| `results/gamma_comparison.png` | Discount-factor comparison |
| `results/epsilon_comparison.png` | Epsilon-decay comparison |
| `results/taxi_trained_agent.mp4` | Video of the trained agent |

Evaluation of the trained policies (30 episodes, exploration off):

| Configuration | Mean reward | Success rate | Mean steps |
|---|---|---|---|
| Baseline (α=0.1, γ=0.99, ε decay=0.995) | 7.63 ± 2.66 | 100% | 13.37 |
| Best (α=0.5, γ=0.99, ε decay=0.99) | 7.63 ± 2.66 | 100% | 13.37 |

Both agents converged to a near-optimal policy. The best configuration reached it much sooner during training (+4.59 average reward at episode 500, versus −95.47 for the baseline).

**Key findings:**

- A low discount factor (γ = 0.5) learned slowest and finished lowest, since the +20 dropoff reward is heavily discounted by the time it reaches states far from the goal.
- Higher learning rates (0.5, 0.9) and faster epsilon decays (0.99, 0.995) mostly made the agent learn faster, not reach a better final policy.

See the PDF report for full discussion.

## Troubleshooting

- **`DeprecatedEnv: Environment version v3 for Taxi is deprecated`** – newer versions of gymnasium replaced Taxi-v3 with Taxi-v4. Either change `"Taxi-v3"` to `"Taxi-v4"` in the code or install an older gymnasium version.
- **Rendering errors about pygame** – make sure gymnasium was installed with the toy-text extras: `pip install "gymnasium[toy-text]"`.