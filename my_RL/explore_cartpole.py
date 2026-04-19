import gymnasium as gym
from cartpole_agent import ActorCriticAgent

def run_sim(num_steps = 1000):
        
    env = gym.make("CartPole-v1")
    agent = ActorCriticAgent(learning_rate= 0.0005, gamma = .99)
    reward_history = []
    state, _ = env.reset()
    total_reward = 0
    done = False

    for i in range(1000):
        action, _, _ = agent.select_action(state)
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
    agent, rewards = run_sim(1000)
    print(f"\nTraining complete!")
    print(f"Final average (last 50): {sum(rewards[-50:])/50:.2f}")




