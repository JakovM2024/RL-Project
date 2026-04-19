import gymnasium as gym
from cartpole_agent import ActorCriticAgent

env = gym.make("CartPole-v1")
agent = ActorCriticAgent(learning_rate= 0.001, gamma = .99)

def train(num_episodes = 2000):
    reward_history = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            agent.store(state, action, log_prob, reward, done, value)

            state = next_state
            total_reward += reward

        agent.update(last_value = 0)
        reward_history.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg_reward = sum(reward_history[-50:]) / 50
            print(action, next_state, reward)
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f}")

    return agent, reward_history

def run_sim(num_steps = 1000):
    reward_history = []
    state, _ = env.reset()
    total_reward = 0
    done = False

    for i in range(1000):
        action, _ , _ = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated

        
        state = next_state
        total_reward += reward
        print(action, next_state, reward)
        
        if done:
            print("FAILED")
            break
    return agent, reward_history

if __name__ == "__main__":
    agent, rewards = train(1000)
    print(f"\nTraining complete!")
    print(f"Final average (last 50): {sum(rewards[-50:])/50:.2f}")
    run_sim(100)


