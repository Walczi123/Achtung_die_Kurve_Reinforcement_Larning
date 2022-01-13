
from game.achtung import Achtung
from game.controllers import DQN_Controller, Man_Controller
from rl.dqn.dqn import get_dqn_model

PLAYER_N = 2

if __name__ == "__main__":
    dqn_controller = DQN_Controller(get_dqn_model())
    players_controllers = [Man_Controller(), dqn_controller]
    game = Achtung(n=PLAYER_N, players_controllers=players_controllers, render_game=True)  
    obs = game.reset()
    game.play()