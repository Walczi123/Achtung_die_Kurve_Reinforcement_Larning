
from v1.game.achtung import Achtung
from v1.game.controllers import CNN_Controller, DQN_Controller, Man_Controller, Random_Controller
from v1.rl.cnn.cnn import get_cnn_model
from v1.rl.dqn.dqn import get_dqn_cnn_model

PLAYER_N = 4

if __name__ == "__main__":
    model = get_dqn_cnn_model()
    # model.load("./rl/dqn/dqn_achtung")
    dqn_controller = DQN_Controller(model)
    # players_controllers = [Man_Controller(), dqn_controller]

    cnn_controller = CNN_Controller(get_cnn_model())
    players_controllers = [cnn_controller, dqn_controller, Random_Controller()]

    game = Achtung(n=len(players_controllers), players_controllers=players_controllers, render_game=True)  
    obs = game.reset()
    winner = game.play()
    print(f'winner {players_controllers[winner].name}')