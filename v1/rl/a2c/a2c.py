import sys
sys.path.append('./../../')
from v1.game.config import WINDOW_HEIGHT, WINDOW_WIDTH
from v1.game.achtung_process import AchtungProcess;
from stable_baselines3 import A2C

def get_a2c_model():
    env = AchtungProcess(n=1, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)

    return A2C("CnnPolicy", 
            env, 
            ent_coef=0.01,
            vf_coef=0.25)
