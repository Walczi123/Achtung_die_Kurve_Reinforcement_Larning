import gym
from stable_baselines3 import DQN
from gym.wrappers import Monitor
import pickle
import os
import time
import matplotlib.pyplot as plt
from gym_achtung.envs.achtungdiekurve import AchtungDieKurve

modelToRun = 'achtung_best_bot.pkl'

def main():
    env = AchtungDieKurve()
    act = DQN("MlpPolicy", env, verbose=1)
    act.learn(total_timesteps=100, log_interval=4)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    outputPath = './Sharks/' + timestr
    os.makedirs(outputPath)

    meanRewards = []
    qValues = []
    numberOfEvaluations = 50
    eval = 0
    while eval < numberOfEvaluations:
        eval += 1
        obs, done = env.reset(), False
        episode_rew = 0
        episode_qVal = []
        while not done:
            env.render()

            getActionQvalue = act.predict(obs)
            action = getActionQvalue[0]

            obs, rew, done, _ = env.step(action)

            episode_rew += rew
            episode_qVal.append(getActionQvalue[1])

        print("Episode reward", episode_rew)
        meanRewards.append(episode_rew)
        qValues.append(episode_qVal)


    outputNameReward = outputPath + '/EnjoyReward.pkl'
    outputNameQvalues = outputPath + '/EnjoyQvalues.pkl'

    with open(outputNameReward, 'wb') as f:
        pickle.dump(meanRewards, f)
        print('Rewards dumped @ ' + outputNameReward )

    with open(outputNameQvalues, 'wb') as f:
        pickle.dump(qValues, f)
        print('Qvalues dumped @ ' + outputNameQvalues)


if __name__ == '__main__':
    main()
