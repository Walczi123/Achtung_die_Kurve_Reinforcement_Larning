
from game.achtung import Achtung

PLAYER_N = 1

if __name__ == "__main__":
    game = Achtung(PLAYER_N, render_game=True)  
    obs = game.reset()
    game.play()