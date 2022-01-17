import pygame
import numpy as np

# game size
WINDOW_HEIGHT = 480
WINDOW_WIDTH = 480
WINDOW_BORDER = 60

# observation
DOWNSCALE = 4
OBS_HEIGHT = int(np.floor(WINDOW_HEIGHT / DOWNSCALE))
OBS_WIDTH = int(np.floor(WINDOW_WIDTH / DOWNSCALE))

# colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
CYAN = (0,255,255)
ORANGE = (255, 69, 0)
RED = (255,0,0)
BLUE = (0,0,255)
COLORS = [MAGENTA,GREEN,CYAN,ORANGE,RED,BLUE]
COLORS_str = ['magenta','green','cyan','orange']

# keys
LEFT_KEYS = [pygame.K_LEFT, pygame.K_a, pygame.K_v, pygame.K_k, pygame.K_n, pygame.K_t]
LEFT_KEYS_str = ['<-', 'a', 'v', 'k', 'n', 't']
RIGHT_KEYS = [pygame.K_RIGHT, pygame.K_s, pygame.K_b, pygame.K_l, pygame.K_m, pygame.K_y]
RIGHT_KEYS_str = ['->','s','b','l', 'm', 'y']

#rewards
LIVING_REWARD = 1
LOSING_REWARD = -1