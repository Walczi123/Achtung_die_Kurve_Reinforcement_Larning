import sys
sys.path.append('./../../')
from v2.gym_achtung.envs.achtungdiekurve import AchtungDieKurve;
from stable_baselines3 import A2C

def get_a2c_cnn_model():
    env = AchtungDieKurve()

    return A2C("CnnPolicy", 
            env, 
            ent_coef=0.01,
            vf_coef=0.25)

def get_a2c_mlp_model():
    env = AchtungDieKurve()

    return A2C("MlpPolicy", 
            env, 
            ent_coef=0.01,
            vf_coef=0.25)