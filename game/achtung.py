import numpy as np
import random
import sys
import os

from game.controllers import Man_Controller
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import pygame.gfxdraw

from math import *
from pygame.locals import *

import gym 
from gym import spaces

from game.config import BLACK, COLORS, LEFT_KEYS, LIVING_REWARD, LOSING_REWARD, RIGHT_KEYS, WINDOW_BORDER, WINDOW_HEIGHT, WINDOW_WIDTH, COLORS_str, LEFT_KEYS_str, RIGHT_KEYS_str 
from game.player import Player

pygame.init()

class Achtung(gym.Env):
    def __init__(self,n=1,id=0, players_controllers:list = None, render_game:bool = False, speed:int = 12, width:int = WINDOW_WIDTH, height:int=WINDOW_HEIGHT):
        # print('Achtung Die Kurve!')
        # pygame.display.set_caption('Achtung Die Kurve!')
        
        # pygame
        self.speed = speed
        self.border = WINDOW_BORDER
        self.window_width = width
        self.window_height = height
        if self.window_height != WINDOW_HEIGHT or self.window_width != WINDOW_WIDTH:
            self.border = min(self.window_height, self.window_width)/4
        self.window_buffer = 1
        self.fps_clock = pygame.time.Clock()
        # print((self.window_width, self.window_height))
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.display = pygame.Surface(self.screen.get_size())
        self.render_game = render_game
        self.cache_frames = False

        # game
        self.game_over = True
        self.first_step = True
        self.n = n
        self.players = self.init_players(n)
        self.players_active = len(self.players)
        self.id = 0
        self.frame = 1
        self.verbose = True
        self.current_player = 0
        self.state_cache = np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8)
        self.players_controllers = self.valid_players_controllers(players_controllers)

        # gym
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.window_width, self.window_height , 3), dtype=np.uint8)
        if self.render_game == False:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        #rewards
        self.living_reward =  LIVING_REWARD
        self.lossing_reward = LOSING_REWARD

        self.reset()

    def valid_players_controllers(self, players_controllers):
        if players_controllers is None or len(players_controllers) != self.n:
            return [Man_Controller() for _ in range(self.n)]
        return players_controllers


    def render(self):
        self.screen.blit(self.display, (0, 0))
        return self.state_cache
    
    def state(self):
        if self.current_player == 0:
            self.state_cache = np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8)
        return self.state_cache.T

    def init_players(self,n):
        # generate players
        players = [Player(self.border) for i in range(n)]
        for i in range(n):
            players[i].gen(self)
            players[i].color = COLORS[i]
            players[i].left_key = LEFT_KEYS[i]
            players[i].right_key = RIGHT_KEYS[i]
        return players

    def reset_color(self):
        self.players[self.current_player].color = COLORS[self.current_player]

    def reset(self):
        self.game_over = False
        self.first_step = True
        self.players_active = self.n
        self.current_player = 0
        self.frame = 0
        self.display.fill(BLACK)
        for i in range(self.n):
            self.players[i].active = True
            self.players[i].gen(self)
            self.players[i].draw(self)
            self.players[i].color = COLORS[i]
        return self.state()

    def close(self):
        pygame.quit()
        sys.exit()

    def check_first_step(self):
        if self.first_step:
            # print('Round %i' % (self.rnd))
            self.first_step = False

    def hole(self):
        hole = random.randrange(1, 20)
        i = self.current_player
        if hole == i+5 and self.players[i].active:
            self.players[i].move()
            self.players[i].color = BLACK

    def update_player(self,action):
        # current player
        i = self.current_player
        player = self.players[i]
        # reset players' colors
        self.reset_color()
        # random hole
        self.hole()
        # action
        if action == 0:
            None
        elif action == 1:            
            player.angle -= 10
        elif action == 2:
            player.angle += 10
        else:
            None
        # update
        if player.active and ((self.players_active > 1 and self.n > 1) or (self.players_active > 0 and self.n == 1)):
            player.angle_reset()
            # checking if someone fails
            if player.collision(self):
                self.players_active -= 1
            player.draw(self)
            player.move()

    def round_over(self):
        if (self.players_active == 1 and self.n > 1) or (self.players_active == 0 and self.n == 1):
            self.game_over = True
            self.id += 1

    def reward(self):
        if self.game_over == False:
            return self.living_reward # nominal reward
        else:
            return self.lossing_reward # losing reward
    
    def to_play(self):
        return self.current_player

    def legal_actions(self):
        return [1, 2, 3]
        
    def step(self,action):
        # current state
        state = self.state()
        # check first step
        self.check_first_step()
        # update current player
        self.update_player(action)
        # check round over
        self.round_over()
        # get reward
        reward = self.reward()
        # check for done
        done = False
        if self.game_over and self.current_player == self.n-1:
            done = True
        # game frames
        if self.current_player == self.n-1:  
            if self.render_game: 
                self.render()
                self.fps_clock.tick(self.speed)
            self.frame += 1
        pygame.display.update()
        # cache frames
        if self.cache_frames:
            if not os.path.exists("./images/game_{}".format(self.id)):
                os.makedirs("./images/game_{}".format(self.id))  
            filename = "./images/game_{}/{}.JPG"
            pygame.image.save(self.display, filename.format(self.id, self.frame))
        # update current player
        self.current_player += 1
        if self.current_player >= self.n:
            self.current_player = 0
        return state, reward, done, {}

    def play(self):
        done = False
        obs = self.state()
        while not done:
            for i in range(self.n):
                action = keyboard_input(self)
                action = self.players_controllers[i].make_move([action, obs])
                obs, reward, done, info = self.step(action)

def valid_number_players(number_of_players:int = 1):
    # input number of players
    _n = number_of_players
    if _n < 5 and _n > 0:
        n = _n
    else:
        print('Invalid number of players, setting to: 2')
        n = 2
    print('  %i players' % (n))
    for i in range(n):
        print('     [%s] (%s,%s)' %
              (COLORS_str[i], LEFT_KEYS_str[i], RIGHT_KEYS_str[i]))
    return n

def keyboard_input(game):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
    action = 0
    i = game.current_player
    keys = pygame.key.get_pressed()
    if keys[game.players[i].left_key]:
        action = 1
    if keys[game.players[i].right_key]:
        action = 2
    return action

def main(argv):
    # get number of players
    n = valid_number_players(argv)

    # setup
    done = True
    game = Achtung(n)
    game.render_game = True
    game.cache_frames = False
    
    obs = game.reset()

    # game
    done = False
    while not done:
        for i in range(game.n):
            # keyboard input
            action = keyboard_input(game)
            # step
            obs, reward, done, info = game.step(action)
            
if __name__ == '__main__':
    main(sys.argv)

