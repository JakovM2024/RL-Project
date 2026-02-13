"""
train.py — The Training Loop
==============================

This file ties everything together and TRAINS the agent.

THE RL TRAINING PROCESS:
1. Create the environment and the agent.
2. Repeat for many EPISODES:
   a. Reset the environment (start a new attempt).
   b. The agent observes the state.
   c. The agent picks an action.
   d. The environment returns the next state, reward, and whether it's done.
   e. The agent learns from this experience (updates its networks).
   f. Repeat steps b-e until the episode ends.
3. Over time, the agent gets better and better at reaching the goal!

WHY MANY EPISODES?
   The agent starts knowing nothing — it moves randomly. Through trial and
   error over hundreds of episodes, it gradually discovers that:
   - Moving toward the goal gives better long-term rewards.
   - Hitting walls is bad.
   - Taking fewer steps is better (because of the -0.1 step penalty).

   This is the magic of RL: the agent figures this out BY ITSELF,
   just from the reward signal. We never tell it "go right then down".
"""

from environment import GridWorld
from agent import ActorCriticAgent
from visualize import plot_rewards, plot_grid_with_path


def train(num_episodes=500, print_every=50):
    """
    Train the actor-critic agent on the grid world.

    Args:
        num_episodes: How many episodes (attempts) to train for.
        print_every: Print progress every N episodes.

    Returns:
        agent: The trained agent.
        env: The environment.
        reward_history: List of total rewards per episode.
    """

    # --- SETUP ---
    env = GridWorld()
    agent = ActorCriticAgent(learning_rate=0.001, gamma=0.99)

    # Track rewards over time to see if the agent is improving
    reward_history = []

    print("Starting Training!")
    print("=" * 50)
    print(f"Grid: {env.rows}x{env.cols} | Episodes: {num_episodes}")
    print(f"Goal: Agent at (0,0) must reach (4,4)")
    print(f"The agent starts knowing NOTHING — watch it learn!\n")

    # --- TRAINING LOOP ---
    for episode in range(num_episodes):

        # Start a new episode — reset the environment
        state = env.reset()
        total_reward = 0  # Track total reward for this episode
        done = False

        # --- EPISODE LOOP ---
        # The agent interacts with the environment step by step
        while not done:

            # 1. Agent observes the state and picks an action
            action, log_prob = agent.select_action(state)

            # 2. Environment responds with next state, reward, and done flag
            next_state, reward, done = env.step(action)

            # 3. Agent learns from this experience
            agent.update(state, reward, next_state, done, log_prob)

            # 4. Move to the next state
            state = next_state
            total_reward += reward

        # Record this episode's total reward
        reward_history.append(total_reward)

        # --- PRINT PROGRESS ---
        if (episode + 1) % print_every == 0:
            # Calculate average reward over the last 'print_every' episodes
            recent_avg = sum(reward_history[-print_every:]) / print_every
            reached_goal = "Yes!" if total_reward > 5 else "No"

            print(f"Episode {episode + 1:>4}/{num_episodes} | "
                  f"Reward: {total_reward:>7.2f} | "
                  f"Avg (last {print_every}): {recent_avg:>7.2f} | "
                  f"Reached Goal: {reached_goal}")

    # --- TRAINING COMPLETE ---
    print("\n" + "=" * 50)
    print("Training Complete!")

    # Show final statistics
    last_50_avg = sum(reward_history[-50:]) / 50
    best_reward = max(reward_history)
    print(f"Average reward (last 50 episodes): {last_50_avg:.2f}")
    print(f"Best episode reward: {best_reward:.2f}")

    # Count how many of the last 50 episodes reached the goal
    goals_reached = sum(1 for r in reward_history[-50:] if r > 5)
    print(f"Goal reached in last 50 episodes: {goals_reached}/50")

    return agent, env, reward_history


def show_learned_path(agent, env):
    """
    Run one episode with the trained agent and record the path it takes.

    After training, we can watch what the agent learned by running it
    through the environment one more time and recording every position.

    Args:
        agent: The trained actor-critic agent.
        env: The grid world environment.

    Returns:
        path: List of (row, col) positions the agent visited.
    """
    state = env.reset()
    path = [env.agent_pos]  # Start with the initial position
    done = False

    print("\n--- Trained Agent's Path ---")
    env.render()

    while not done:
        action, _ = agent.select_action(state)
        state, reward, done = env.step(action)
        path.append(env.agent_pos)

        # Print each step
        action_name = env.action_names[action]
        print(f"Action: {action_name} → Position: {env.agent_pos} "
              f"(Reward: {reward:.1f})")

    if env.agent_pos == env.goal:
        print(f"\nGoal reached in {len(path) - 1} steps!")
    else:
        print(f"\nFailed to reach goal after {len(path) - 1} steps.")

    return path


# --- MAIN ---
if __name__ == "__main__":
    # Train the agent
    agent, env, reward_history = train(num_episodes=500)

    # Show the path the trained agent takes
    path = show_learned_path(agent, env)

    # Visualize the results
    print("\nGenerating plots...")
    plot_rewards(reward_history)
    plot_grid_with_path(env, path)
    print("Done! Check the plot windows.")
