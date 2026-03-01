import gymnasium as gym

env = gym.make("CartPole-v1")

state, _ = env.reset()
print(state)
for i in range(100):
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    print(action, next_state, reward)
    if done:
        print("FAILED")
        break



