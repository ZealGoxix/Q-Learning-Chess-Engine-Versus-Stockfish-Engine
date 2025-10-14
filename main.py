import gym
import numpy as np
import time  # Add this import for delays

# 1. Create the environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# 2. Initialize the Q-table (16 states, 4 actions)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# 3. Set hyperparameters
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 0.1
episodes = 10000

# 4. Training loop
for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        # 4a. Choose action: Explore or Exploit?
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state]) # Exploit

        # 4b. Take the action
        next_state, reward, done, info = env.step(action)

        # 4c. Update Q-table using the Q-learning formula
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state

print("Training finished!\n")

# 5. Evaluate the trained agent (NEW CODE)
print("Evaluating trained agent...")
test_episodes = 10
success_count = 0

for episode in range(test_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    print(f"\nTest Episode {episode + 1}")
    print("Starting state:")
    env.render()
    
    step_count = 0
    while not done:
        # Choose the best action based on learned Q-table (no exploration)
        action = np.argmax(q_table[state])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step_count}: Action {action}", end=" ")
        if action == 0: print("(LEFT)", end=" ")
        elif action == 1: print("(DOWN)", end=" ")
        elif action == 2: print("(RIGHT)", end=" ")
        elif action == 3: print("(UP)", end=" ")
        print(f"- Reward: {reward}")
        
        state = next_state
        step_count += 1
        
        # Small delay to see what's happening
        time.sleep(0.5)
    
    if reward == 1:  # Successfully reached goal
        success_count += 1
        print("✅ SUCCESS: Reached the goal!")
    else:
        print("❌ FAILED: Fell in a hole!")
    
    print(f"Total reward: {total_reward}")

print(f"\nSuccess rate: {success_count}/{test_episodes} ({(success_count/test_episodes)*100}%)")

# 6. Visualize the final Q-table (NEW CODE)
print("\nFinal Q-table:")
print("State | Left     Down     Right    Up")
print("-" * 40)
for state in range(16):
    print(f"{state:5} | {q_table[state, 0]:.4f}  {q_table[state, 1]:.4f}  {q_table[state, 2]:.4f}  {q_table[state, 3]:.4f}")

# 7. Watch the agent play with visualization (NEW CODE)
input("\nPress Enter to watch the trained agent play with visualization...")

# Recreate environment with human rendering
env.close()
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state = env.reset()
done = False

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, info = env.step(action)
    time.sleep(0.8)  # Slow down to see movements

env.close()