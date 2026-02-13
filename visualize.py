"""
visualize.py — Visualization Tools
=====================================

This file provides visual tools to understand what the agent learned.

Visualizations are crucial in RL because:
1. The REWARD CURVE shows whether the agent is actually learning over time.
   A good curve starts low (random behavior) and trends upward (learned behavior).
2. The GRID PATH shows the actual route the trained agent takes,
   so you can see if it found an efficient path to the goal.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_rewards(reward_history, window=20):
    """
    Plot the reward per episode over training.

    WHAT TO LOOK FOR:
    - Early episodes: Low rewards (the agent is moving randomly).
    - Middle episodes: Rewards start climbing (the agent is learning).
    - Late episodes: Rewards plateau near the maximum (the agent has converged).

    The smoothed line (moving average) makes it easier to see the trend
    since individual episodes can be noisy.

    Args:
        reward_history: List of total rewards per episode.
        window: Size of the moving average window for smoothing.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = range(1, len(reward_history) + 1)

    # Plot raw rewards (semi-transparent so we can see the noise)
    ax.plot(episodes, reward_history, alpha=0.3, color="blue", label="Raw reward")

    # Plot smoothed rewards (moving average)
    if len(reward_history) >= window:
        smoothed = []
        for i in range(len(reward_history)):
            start = max(0, i - window + 1)
            avg = sum(reward_history[start:i + 1]) / (i - start + 1)
            smoothed.append(avg)
        ax.plot(episodes, smoothed, color="red", linewidth=2,
                label=f"Smoothed (window={window})")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Training Progress: Reward per Episode", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add a horizontal line at reward=10 (perfect score minus some step costs)
    ax.axhline(y=8, color="green", linestyle="--", alpha=0.5, label="Good performance")

    plt.tight_layout()
    plt.savefig("training_rewards.png", dpi=150)
    print("Saved: training_rewards.png")
    plt.show()


def plot_grid_with_path(env, path):
    """
    Draw the grid world with the agent's path overlaid.

    This shows you the actual route the trained agent takes from
    start (S) to goal (G). A well-trained agent should take a
    short, direct path avoiding walls.

    Args:
        env: The GridWorld environment.
        path: List of (row, col) positions the agent visited.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw the grid cells
    for row in range(env.rows):
        for col in range(env.cols):
            # Choose color based on cell type
            if (row, col) in env.walls:
                color = "#2c3e50"      # Dark gray for walls
            elif (row, col) == env.start:
                color = "#3498db"      # Blue for start
            elif (row, col) == env.goal:
                color = "#2ecc71"      # Green for goal
            else:
                color = "#ecf0f1"      # Light gray for empty

            # Draw the cell as a colored square
            rect = patches.Rectangle((col, env.rows - 1 - row), 1, 1,
                                     linewidth=1, edgecolor="white",
                                     facecolor=color)
            ax.add_patch(rect)

            # Add labels
            if (row, col) == env.start:
                ax.text(col + 0.5, env.rows - 1 - row + 0.5, "S",
                        ha="center", va="center", fontsize=16,
                        fontweight="bold", color="white")
            elif (row, col) == env.goal:
                ax.text(col + 0.5, env.rows - 1 - row + 0.5, "G",
                        ha="center", va="center", fontsize=16,
                        fontweight="bold", color="white")
            elif (row, col) in env.walls:
                ax.text(col + 0.5, env.rows - 1 - row + 0.5, "X",
                        ha="center", va="center", fontsize=16,
                        fontweight="bold", color="white")

    # Draw the agent's path as arrows
    if len(path) > 1:
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            # Convert grid coordinates to plot coordinates
            x1 = c1 + 0.5
            y1 = env.rows - 1 - r1 + 0.5
            x2 = c2 + 0.5
            y2 = env.rows - 1 - r2 + 0.5

            # Draw arrow from current to next position
            dx = x2 - x1
            dy = y2 - y1

            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color="#e74c3c",
                                        lw=2.5))

            # Number each step
            ax.text(x1, y1 + 0.3, str(i + 1), ha="center", va="center",
                    fontsize=8, color="#e74c3c", fontweight="bold")

    # Set up the plot
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.set_title(f"Agent's Learned Path ({len(path) - 1} steps)", fontsize=14)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("agent_path.png", dpi=150)
    print("Saved: agent_path.png")
    plt.show()


# --- QUICK TEST ---
# If you run this file directly, it shows an example with a random path
if __name__ == "__main__":
    from environment import GridWorld

    print("Testing visualizations with a random path...")

    env = GridWorld()
    # Fake path for testing
    test_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3),
                 (2, 4), (3, 4), (3, 3), (3, 2), (4, 2), (4, 3), (4, 4)]

    # Fake reward history
    test_rewards = [r + np.random.randn() * 2 for r in np.linspace(-5, 9, 100)]

    plot_rewards(test_rewards)
    plot_grid_with_path(env, test_path)
