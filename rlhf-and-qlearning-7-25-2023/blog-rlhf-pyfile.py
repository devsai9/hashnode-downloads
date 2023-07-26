# Code written by: Sai Siddhish Chandra Sekaran (https://hashnode.com/@Cakez)

# It is highly recommended to run the ipynb file instead of this py file as there are multiple print statements and it could get confusing

# ------------------------------------------------------------

# MAKE SURE TO RUN THE FOLLOWING COMMANDS IN YOUR TERMINAL:
# pip install numpy
# pip install gymnasium

# Python code to import the libraries we will use later
import numpy as np
import gymnasium as gym

# Make the environment
env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="ansi", is_slippery=False)

# Reset the environment to get a observation
observation, info = env.reset()

# Printing the state
# When the observation is printed, you will see that we recieve the number 0 like mentioned above
print(observation)

print(env.render())
# S is the starting point
# G is the goal
# H's are holes
# F's are the Frozen lake
# The letter with a red/pink background is the character's current position

# Now we will take a random action in the environment and see what we get in response

# Selecting a random action
action = env.action_space.sample()

# Executing that random action in the environment
observation, reward, terminated, truncated, info = env.step(action)

print(f"Observation: {observation}")
print(f"Reward: {reward}")
print(f"Environment Terminated?: {terminated}")
print(f"Environment Truncated?: {truncated}")

# Render the game
print("\n" + env.render())

# The character moving on it's own is simply random, next we will implement a RL model to make it play better!

# Initialize our Q-Table to all 0 values (the RL agent bases it's decisions off of this table and updates it in training)
QTable = np.zeros((env.observation_space.n, env.action_space.n))

# Greedy Epsilon Function, this is what determines what action we will take
def greedyEpsilon(Qtable, state, epsilon):
  num = np.random.rand()
  if (num < epsilon):
    action = env.action_space.sample()
  else:
    action = np.argmax(Qtable[state])
  return action

# Now we will train our agent (this might take a few minutes)
def train(env, QTable, numOfEpisodes, learningRate, discountFactor, startingEpsilon, finalEpsilon, decayRate, maxSteps):
  for i in range(numOfEpisodes):
    # Decaying Epsilon so that we get more exploitation than exploration (over time)
    epsilon = startingEpsilon + (finalEpsilon - startingEpsilon) * np.exp(-decayRate * i)

    # Reset the environment and get an observation
    currentState = env.reset()
    currentState = currentState[0]

    for j in range(maxSteps):
      # Determine what action to take with the Greedy Epsilon function
      action = greedyEpsilon(QTable, currentState, epsilon)

      # Retrieve important info from environment and apply our action in the environment
      newState, reward, terminated, truncated, info = env.step(action)

      # Update the Q Table to reflect what the agent has learned
      QTable[currentState][action] = (1 - learningRate) * QTable[currentState][action] + learningRate * (reward + discountFactor * np.max(QTable[newState]))

      # If the game is terminated or truncated, finish this session (this "session" is often called an "episode")
      if (terminated or truncated):
        break

      # Update the state we were basing everything off of, to the new state
      currentState = newState
  return QTable

train(env, QTable, 10000, 0.5, 0.97, 0.01, 1, 0.0005, 10000)

# Lets see how well our agent performs when playing the game

# Variables
numOfWins = 0
numOfLosses = 0
for i in range(100):
  # Get the current state/observation
  state = env.reset()
  state = state[0]
  for j in range(1000):
    # Get the action from the QTable with the highest reward
    action = np.argmax(QTable[state][:])

    # Retrieve important info from environment and apply our action in the environment
    newState, reward, terminated, truncated, info = env.step(action)

    # If the reward equals 1 (we got to the goal) add one to the number of wins
    # Else add one to the number of losses
    if (reward == 1):
      numOfWins += 1
    else:
      numOfLosses += 1

    # If the game is terminated or truncated, finish this session (this "session" is often called an "episode")
    if (terminated or truncated):
        break

    # Update the state we were basing everything off of, to the new state
    state = newState

print(f"Number of wins: {numOfWins}")
print(f"Number of losses: {numOfLosses}")


# Thank you for reading the article and running this code!<br>
# Also consider following me over on Hashnode at https://hashnode.com/@Cakez. All the support is appreciated