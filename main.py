import gymnasium as gym
import numpy as np
import time  # For adding delays during visualization

# 1. Create the environment with modern Gymnasium
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

# 2. Initialize the Q-table (16 states, 4 actions)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# 3. Set hyperparameters
learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 0.1
episodes = 10000

print("Starting training...")

# 4. Training loop
for episode in range(episodes):
    # Reset returns (state, info) - we need to unpack both
    state, info = env.reset()
    done = False

    while not done:
        # 4a. Choose action: Explore or Exploit?
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # 4b. Take the action - Gymnasium returns 5 values now
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # In Gymnasium, 'done' is split into 'terminated' and 'truncated'
        done = terminated or truncated

        # 4c. Update Q-table using the Q-learning formula
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        # Q-learning formula
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state

env.close()
print("Training finished!\n")

# 5. Evaluate the trained agent
print("Evaluating trained agent...")
test_episodes = 5
success_count = 0

for episode in range(test_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    print(f"\nTest Episode {episode + 1}")
    
    while not done:
        # Always choose the best action based on learned Q-table
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        state = next_state
        step_count += 1
    
    if reward == 1:  # Successfully reached goal
        success_count += 1
        print(f"✅ SUCCESS: Reached the goal in {step_count} steps!")
    else:
        print(f"❌ FAILED: Fell in a hole after {step_count} steps")
    
    print(f"Total reward: {total_reward}")

print(f"\nSuccess rate: {success_count}/{test_episodes}")

# 6. Visualize the agent playing with human rendering
print("\nPreparing visualization...")
time.sleep(2)

# Recreate environment with human rendering
env.close()
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

state, info = env.reset()
done = False

print("Watch the trained agent play...")
time.sleep(1)

while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    time.sleep(0.5)  # Slow down to see the movements

env.close()

# 7. Display the learned Q-table
print("\nLearned Q-table (first 5 states):")
print("State | Left     Down     Right    Up")
print("-" * 40)
for state in range(5):  # Show first 5 states
    print(f"{state:5} | {q_table[state, 0]:.4f}  {q_table[state, 1]:.4f}  {q_table[state, 2]:.4f}  {q_table[state, 3]:.4f}")