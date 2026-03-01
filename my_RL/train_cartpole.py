import gymnasium as gym
from cartpole_agent import ActorCriticAgent

def train(num_episodes = 1000):
    env = gym.make('CartPole-v1')

    agent = ActorCriticAgent(learning_rate= 0.0005, gamma = .99)
    reward_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            agent.update(state, reward, next_state, done, log_prob)

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg_reward = sum(reward_history[-50:]) / 50
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f}")

    return agent, reward_history

if __name__ == "__main__":
    agent, rewards = train(1000)
    print(f"\nTraining complete!")
    print(f"Final average (last 50): {sum(rewards[-50:])/50:.2f}")


