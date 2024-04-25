from stable_baselines3 import PPO, A2C
import os
from EldenEnv import EldenEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces



#    def __init__(self, observation_space:spaces.Box, features_dim: int = 512):
#        super().__init__(observation_space,features_dim)
#        # Define CNN for image processing
#        n_input_channels = observation_space.spaces['img'].shape[0]
#        self.cnn = nn.Sequential(
#            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
#            nn.ReLU(),
#            nn.Conv2d(32, 64, kernel_size=4, stride=2),
#            nn.ReLU(),
#            nn.Conv2d(64, 64, kernel_size=3, stride=1),
#            nn.ReLU(),
#            nn.Flatten(),
#        )
#        
#        # Calculate CNN output size dynamically
#        self.cnn_output_size = self._get_cnn_output(observation_space)
#        
#        # Flatten prev_actions size
#        prev_actions_size = np.prod(observation_space.spaces['prev_actions'].shape).item()
#        state_size = np.prod(observation_space.spaces['state'].shape).item()
#
#        # Combined feature size
#        self.combined_feature_size = self.cnn_output_size + prev_actions_size + state_size
#
#    def _get_cnn_output(self, observation_space):
#        with torch.no_grad():
#            sample_input = torch.as_tensor(observation_space.spaces['img'].sample()[None]).float()
#            #sample_input = sample_input.permute(0, 3, 1, 2)  # Change to (N, C, H, W)
#            sample_output = self.cnn(sample_input)
#            return int(np.prod(sample_output.size()))
#
#    def forward(self, observations):
#        img_input = observations['img']#.permute(0, 3, 1, 2)
#        cnn_features = self.cnn(img_input)
#
#        prev_actions_flat = torch.flatten(observations['prev_actions'], start_dim=1).float()
#        state_features = observations['state'].float()
#
#        combined_features = torch.cat((cnn_features, prev_actions_flat, state_features), dim=1)
#        return combined_features
def set_affinity(cores):
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cores)  # cores is a list of core IDs, e.g., [0, 1]

class CustomFeatureExtractor(BaseFeaturesExtractor):
    set_affinity([0, 1])
    """
    Custom feature extractor for handling image, previous actions, and state observations.

    :param observation_space: (gym.spaces.Dict) The observation space consisting of
        an image, previous actions, and a state.
    :param features_dim: (int) Number of features extracted for the last layer.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Image observation
        n_input_channels = observation_space.spaces['img'].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flat size by doing one forward pass with a dummy tensor
        with torch.no_grad():
            # Total features size after concatenation
            total_features_size = 135054

            # Final linear layer
        self.linear = nn.Sequential(nn.Linear(total_features_size, features_dim), nn.ReLU())

    def forward(self, observations: dict) -> torch.Tensor:
        # Process image observation through CNN
        image_features = self.cnn(observations['img'])

        # Flatten prev_actions and state (if not already flat)
        prev_actions_features = observations['prev_actions']
        state_features = observations['state']
        # Flatten image_features if not already
        prev_actions_features = prev_actions_features.view(prev_actions_features.size(0), -1)  # Ensure 2D
        state_features = state_features.view(state_features.size(0), -1)

        # Concatenate all features
        combined_features = torch.cat([image_features, prev_actions_features, state_features], dim=1)
        
        # Pass through the final linear layer
        return self.linear(combined_features)

def train(CREATE_NEW_MODEL, config):
    set_affinity(list(range(16, 30)))
    print("🧠 Training will start soon. This can take a while to initialize...")
    
    
    TIMESTEPS = 1024			#Learning rate multiplier.
    HORIZON_WINDOW = 1024	#Lerning rate number of steps before updating the model. ~2min


    '''Creating folder structure'''
    model_name = "PPO-1" 					
    if not os.path.exists(f"models/{model_name}/"):
        os.makedirs(f"models/{model_name}/")
    if not os.path.exists(f"logs/{model_name}/"):
        os.makedirs(f"logs/{model_name}/")
    models_dir = f"models/{model_name}/"
    logdir = f"logs/{model_name}/"			
    model_path = f"{models_dir}/PPO-1"
    print("🧠 Folder structure created...")

    
    '''Initializing environment'''
    env = EldenEnv(config)
    print("🧠 EldenEnv initialized...")

    policy_kwargs = dict(
    features_extractor_class=CustomFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=512)  # Specify the expected features_dim if needed
    #net_arch=[128,dict(pi=[64, 28, 14], vf=[64, 32, 16])]  # Example architecture for actor (pi) and critic (vf)
)
    '''Creating new model or loading existing model'''
    if CREATE_NEW_MODEL:
        model = PPO('MultiInputPolicy',
							env,
							tensorboard_log=logdir,
							n_steps=HORIZON_WINDOW,
							batch_size=128,
							verbose=1,
							n_epochs=1,
							policy_kwargs=policy_kwargs,
                            ent_coef=0.01,
							device='auto')	#Set training device here.
        print("🧠 New Model created...")
    else:
        model = PPO.load(model_path, env=env)
        print("🧠 Model loaded...")
		


    '''Training loop'''
    i=0
    while i<4:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", log_interval=1)
        model.save(f"{models_dir}/PPO-1")
        torch.cuda.empty_cache()
        print(f"🧠 Model updated...")
        i=i+1