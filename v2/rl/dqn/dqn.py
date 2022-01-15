import sys
sys.path.append('./../../../')
from v2.gym_achtung.envs.achtungdiekurve import AchtungDieKurve;
from stable_baselines3 import DQN

def get_dqn_cnn_model():
    env = AchtungDieKurve()
    return DQN("CnnPolicy", 
            env, 
            buffer_size=100,
            learning_rate=1e-4,
            batch_size=100,
            learning_starts=100000,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.01)

def get_dqn_mlp_model():
    env = AchtungDieKurve()
    return DQN("MlpPolicy", 
            env, 
            buffer_size=100,
            learning_rate=1e-4,
            batch_size=100,
            learning_starts=100000,
            target_update_interval=1000,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.1,
            exploration_final_eps=0.01)
