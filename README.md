This ia a repo using reinforcement learning to play Elden Ring, modified from https://github.com/ocram444/EldenRL. 
The code use openai gym for custom environment, opencv for grabbing the frames of the game and stable_baselines3 PPO algorithm as the main model.
The process of each step is :

  1.grab the current screen for player health and boss health, then use these variables to calculate the reward of the previous action.
  2.grab frames from the game window(4 continuous frames to show motion of the boss)
  3.player act according to the frames input, the player health and its previous 10 actions(oneHot endcoded)
  
also some modification was made compared to ocram444's original version:
  1. use CNN to process image input(2 convolutional layer)
  2. change reward function
Now Im trying to use LLM to optimize the reward function and ways to improve the RL network structure

Questions:
  1. Need applicable means to optimize the reward function using LLLM
  2. How to set the CNN and actor-critic network in PPO to gain better performance
  3. Need tricks fine-tuning the PPO
