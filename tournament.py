import matplotlib.pyplot as plt
from v1.game.achtung import Achtung
from v1.game.controllers import CNN_Controller, DQN_Controller, Man_Controller, Random_Controller, A2C_Controller
from v1.rl.cnn.cnn import get_cnn_model
from v1.rl.dqn.dqn import get_dqn_cnn_model, get_dqn_mlp_model
from v1.rl.a2c.a2c import get_a2c_cnn_model, get_a2c_mlp_model

GAMES = 1000

if __name__ == "__main__":
    dqn_cnn = get_dqn_cnn_model()
    dqn_cnn.load("./achtung_tests/models/dqn_cnn_v1.zip")
    dqn_mlp = get_dqn_mlp_model()
    dqn_mlp.load("./achtung_tests/models/dqn_mlp_v1.zip")
    a2c_cnn = get_a2c_cnn_model()
    a2c_cnn.load("./achtung_tests/models/a2c_cnn_v1.zip")
    a2c_mlp = get_a2c_mlp_model()
    a2c_mlp.load("./achtung_tests/models/a2c_mlp_v1.zip")
    cnn = get_cnn_model()
    cnn.load("./achtung_tests/models/cnn_v1")

    print('correct loaded')

    players_controllers = [CNN_Controller(cnn), DQN_Controller(dqn_cnn), DQN_Controller(dqn_mlp), A2C_Controller(a2c_mlp), A2C_Controller(a2c_cnn), Random_Controller()]
    players_names = ['cnn', 'dqn_cnn', 'dqn_mlp', 'a2c_mlp', 'a2c_cnn', 'random']
    winners=dict()
    for name in players_names:
        winners[name] = 0
    
    for i in range(GAMES):
        game = Achtung(n=len(players_controllers), players_controllers=players_controllers, render_game=False)  
        obs = game.reset()
        winner = game.play()
        print(f'game no. {i+1} winner {winner+1}')
        winners[players_names[winner]] += 1

    plt.bar(players_names, list(winners.values()))
    plt.xlabel('algorithm')
    plt.ylabel('no. wins')
    plt.title('Tournament')
    plt.savefig(f'./achtung_tests/plots/tournament1.png')