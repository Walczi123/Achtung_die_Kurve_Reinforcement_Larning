import numpy as np
import pickle
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

ITERATIONS = 10 #150
LEARN_STEP = 10 #100
EVALUATE_POLICY_EPISODES = 10 #100

def test_and_save(paramas):
    (model, model_name) = paramas
    rewards = []
    stds = []
    for i in range(ITERATIONS):
        print("iteration: ", i+1)
        model.learn(total_timesteps=LEARN_STEP)
        mean_reward, std_reward = evaluate_policy(model, model.env, n_eval_episodes=EVALUATE_POLICY_EPISODES)
        print(f"   mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        
        rewards.append(mean_reward)
        stds.append(std_reward)
        
    print("   saving...")

    with open(f"./tests/results/test_{model_name}_reward", "wb") as f:   
        pickle.dump(rewards, f)
    with open(f"./tests/results/test_{model_name}_std", "wb") as f:   
        pickle.dump(stds, f)

    model.save(f"./tests/models/{model_name}")

    print("   saved")

def read_and_show_graph(model_name):
    with open(f"./tests/results/test_{model_name}_reward", "rb") as f:   
        rewards = np.array(pickle.load(f))
    with open(f"./tests/results/test_{model_name}_std", "rb") as f:   
        stds = np.array(pickle.load(f))

    # fig, ax = plt.figure()
    plt.plot(rewards)
    plt.xlabel('epoch')
    plt.ylabel('episode reward')
    plt.title(model_name)

    my_xticks = [x for x in range(LEARN_STEP, (ITERATIONS+1)*LEARN_STEP, ITERATIONS)]
    print(my_xticks)
    plt.xticks(range(len(my_xticks)), my_xticks)

    plt.fill_between(range(len(rewards)),rewards-stds,rewards+stds,alpha=.3)
    plt.savefig(f'./tests/plots/{model_name}.png')