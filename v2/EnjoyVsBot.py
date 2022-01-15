import gym
import gym_achtung
from stable_baselines3 import DQN
import time

from gym.utils import play

from gym_achtung.envs.achtungdiekurveAgainstBot import AchtungDieKurveAgainstBot

model_to_run = "achtung_best_bot.pkl"

def main():
    env = AchtungDieKurveAgainstBot()
    act = DQN("MlpPolicy", env, verbose=1)
    act.learn(total_timesteps=100, log_interval=4)
    number_of_evaluations = 50
    eval = 0
    while eval < number_of_evaluations:
        eval += 1
        obs, done = env.reset(), False
        print(eval)
        while not done:
            time.sleep(0.01)
            env.render()
            getActionQvalue = act.predict(obs)
            action = getActionQvalue[0]
            obs, rew, done, _ = env.step(action)


if __name__ == '__main__':
    main()
