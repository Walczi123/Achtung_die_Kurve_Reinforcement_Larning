
from game.achtung import Achtung
from game.controllers import CNN_Controller, DQN_Controller, Man_Controller, Random_Controller
from rl.cnn.cnn import select_action
from rl.dqn.dqn import get_dqn_model

PLAYER_N = 4

if __name__ == "__main__":
    model = get_dqn_model()
    model.load("./rl/dqn/dqn_achtung")
    dqn_controller = DQN_Controller(model)
    # players_controllers = [Man_Controller(), dqn_controller]

    cnn_controller = CNN_Controller(select_action)
    players_controllers = [Man_Controller(), cnn_controller, dqn_controller, Random_Controller()]

    game = Achtung(n=len(players_controllers), players_controllers=players_controllers, render_game=True)  
    obs = game.reset()
    winner = game.play()
    print(f'winner {players_controllers[winner].name}')