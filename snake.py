#!/usr/bin/env python3

import pygame
from pygame.locals import *
from enum import Enum
import random
import numpy as np

clock = pygame.time.Clock()

pygame.init()

# Color constance
color_background = (255, 255, 255)
color_snake = (50, 50, 50)
color_block = (255, 50, 50)

# Size constance
grid_step = 20
grid_size = (25, 20)
surface_size = (grid_step * grid_size[0], grid_step * grid_size[1])



def random_block():
    pos = (random.randint(0, grid_size[0]-1) * grid_step + 1, random.randint(0, grid_size[1]-1) * grid_step + 1)
    return pygame.Rect(pos, (grid_step - 2, grid_step - 2))

class NeuralAI:
    n_input = 12
    n_hidden = 8
    n_output = 4

    def __init__(self):
        self.layer1 = np.random.rand(self.n_input, self.n_hidden) * 2 - 1
        self.layer2 = np.random.rand(self.n_hidden, self.n_output) * 2 - 1
        self.bias1 = np.random.rand(self.n_hidden) * 2 - 1
        self.bias2 = np.random.rand(self.n_output) * 2 - 1

    def predict(self, input_data):
        l1 = self.relu(np.dot(input_data.transpose(), self.layer1) + self.bias1)
        l2 = self.relu(np.dot(l1, self.layer2) + self.bias2)
        return l2

    def relu(self, x):
        return np.maximum(0, x) 


class Player:

    def __init__(self, ai):
        self.body = [(pygame.Rect((random.randint(1, grid_size[0] - 1) * grid_step + 1, random.randint(1, grid_size[1]) * grid_step + 1), (grid_step - 2, grid_step - 2)))]
        #self.body = [(pygame.Rect((1, 1), (grid_step - 2, grid_step - 2)))]
        self.direction = 2
        self.eating = []
        self.body.append
        if ai is None:
            self.ai = NeuralAI()
        else:
            self.ai = ai
        self.tick = 0
        self.killed = False

    def update(self):
        add_block = False
        if len(self.eating) != 0 and self.body[-1].contains(self.eating[0]):
            add_block = True

        for i in range(len(self.body)-1, 0, -1):
            self.body[i] = self.body[i-1].copy()
                
        if self.direction == 0:
            self.body[0].move_ip(0, -grid_step)
        elif self.direction == 1:
            self.body[0].move_ip(grid_step, 0)
        elif self.direction == 2:
            self.body[0].move_ip(0, grid_step)
        elif self.direction == 3:
            self.body[0].move_ip(-grid_step, 0)

        if add_block:
            self.body.append(self.eating.pop(0))

        self.tick +=1
        if self.tick > 80* len(self.body):
            self.killed = True

    def draw(self, surface):
        for i in range(0,len(self.body)):
            pygame.draw.rect(surface, color_snake, self.body[i])

    def score(self):
        #if self.killed:
        #    return 0
        return len(self.body)

class Game:
    def __init__(self, player):
        self.block = random_block()
        self.player = player
        self._tick = 0
        self.isRunning = True
    
    def update(self):
        # Predict action
        predict = self.player.ai.predict(self.distance())
        predict[(self.player.direction + 2) % 4] = 0
        self.player.direction = np.argmax(predict)

        self.player.update()
        if self.player.body[0].contains(self.block):
            self.player.eating.append(self.block.copy())
            self.block = random_block()
        if self.player.body[0].x < 0 or self.player.body[0].x > surface_size[0] or self.player.body[0].y < 0 or self.player.body[0].y > surface_size[1]:
            self.isRunning = False
            #print("Mur atteint")
        if self.player.body[0].collidelist(self.player.body[1:]) != -1:
            self.isRunning = False
            #print("Perdu") 
        if self.player.killed:
            self.isRunning = False

    def distance(self):
        l1 = [(self.player.body[0].x - 1) / grid_step,
            (surface_size[0] - (self.player.body[0].x - 1)) / grid_step - 1,
            (self.player.body[0].y - 1) / grid_step,
            (surface_size[1] - (self.player.body[0].y - 1)) / grid_step - 1]
        l2 = [(self.player.body[0].x - self.block.x) /grid_step,
            (self.player.body[0].y - self.block.y) / grid_step]
        l2 = [30 if l2[0] >= 0 else 0,
            30 if l2[1] >= 0 else 0,
            30 if l2[0] <= 0 else 0,
            30 if l2[1] <= 0 else 0]
        l_body = [(self.player.body[i].x, self.player.body[i].y) for i in range(1, len(self.player.body))]
        l_body.sort()
        if l_body is None:
            l3 = [30, 30, 30, 30]
        else:
            l_body_up = list(filter(lambda c: self.player.body[0].x == c[0] and self.player.body[0].y > c[0], l_body))
            l_body_right = list(filter(lambda c: self.player.body[0].y == c[1] and self.player.body[0].x < c[0], l_body))
            l_body_down = list(filter(lambda c: self.player.body[0].x == c[0] and self.player.body[0].y < c[1], l_body))
            l_body_left = list(filter(lambda c: self.player.body[0].y == c[1] and self.player.body[0].x > c[0], l_body))
            l3 = [30 if l_body_up == [] else ((self.player.body[0].y - l_body_up[0][1]) / grid_step-1), 
                30 if l_body_right == [] else - ((self.player.body[0].x - l_body_right[0][0]) / grid_step-1), 
                30 if l_body_down == [] else - ((self.player.body[0].y - l_body_down[0][1]) / grid_step-1), 
                30 if l_body_left == [] else ((self.player.body[0].x - l_body_left[0][0]) / grid_step-1)]

        return np.concatenate([l1, l2, l3])
        #print((self.body[0].x - 1) / grid_step)
        #print((self.body[0].x - 1) / grid_step)
        #print((self.body[0].x - 1) / grid_step)
        #print((self.body[0].x - 1) / grid_step)

    def draw(self, surface):
        # Screen
        surface.fill(color_background)
        # Block
        pygame.draw.rect(surface, color_block, self.block)
        # Player
        self.player.draw(surface)

        pygame.display.flip()

    def loop(self):
        if self.isRunning:
            self.update()
            

class Population:
    n_best = 4
    def __init__(self, n_population):
        self.n_population = n_population
        self.players = [Player(None) for i in range(n_population)]
        self.generator = []

    def selectBest(self):        
        scores = [self.players[i].score() for i in range(self.n_population)]
        ind = np.argpartition(scores, -self.n_best)[-self.n_best:]

        self.generator = list(np.array(self.players)[ind])

    def crossover(self):
        self.players = [Player(p.ai) for p in self.generator]

        while len(self.players) < self.n_population:
            a = self.players[random.randint(0, len(self.players) -1)]
            b = self.players[random.randint(0, len(self.players) -1)]

            c = Player(None)

            layer1_rnd = np.random.randint(2, size=(NeuralAI.n_input, NeuralAI.n_hidden))
            c.ai.layer1 = a.ai.layer1 * layer1_rnd + b.ai.layer1 * (np.ones((NeuralAI.n_input, NeuralAI.n_hidden)) - layer1_rnd)

            layer2_rnd = np.random.randint(2, size=(NeuralAI.n_hidden, NeuralAI.n_output))
            c.ai.layer2 = a.ai.layer2 * layer2_rnd + b.ai.layer2 * (np.ones((NeuralAI.n_hidden, NeuralAI.n_output)) - layer2_rnd)

            bias1_rnd = np.random.randint(2, size=(NeuralAI.n_hidden))
            c.ai.bias1 = a.ai.bias1 * bias1_rnd + b.ai.bias1 * (np.ones((NeuralAI.n_hidden)) - bias1_rnd)
            
            bias2_rnd = np.random.randint(2, size=(NeuralAI.n_output))
            c.ai.bias2 = a.ai.bias2 * bias2_rnd + b.ai.bias2 * (np.ones((NeuralAI.n_output)) - bias2_rnd)

            self.players.append(c)

    def mutate(self):
        for i in np.random.randint(n_population, size=int(n_population*0.05)) : # 10% de la pop
            self.players[i] = Player(None)


def create_surface(i):
    y, x = i // 3, i % 3
    return screen.subsurface(Rect((x * surface_size[0] + x, y * surface_size[1] + y), surface_size))

if __name__ == "__main__" :
    n_surface = 6
    screen_size = (surface_size[0] * 3 + 2, surface_size[1] * 2 + 1)
    screen = pygame.display.set_mode(screen_size)
    surfaces = [create_surface(i) for i in range(n_surface)]    

    n_population = 2000
    generation = 0

    population = Population(n_population)
    games = [Game(population.players[i]) for i in range(n_population)]

    isAppRunning = True
    while isAppRunning:
        generationRunning = any([games[i].isRunning for i in range(n_population)])
        remaining = list(filter(lambda game: game.isRunning, games))
        clock.tick(30)
        for i in range(n_population):
            games[i].loop()
        for i in range(min(n_surface, len(remaining))):
            remaining[i].draw(surfaces[i])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isAppRunning = False
            if event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    isAppRunning = False
        if not generationRunning:
            print(sum([population.players[i].score() for i in range(n_population)])/n_population)
            population.selectBest()
            population.crossover()
            population.mutate()
            games = [Game(population.players[i]) for i in range(n_population)]
            generation += 1
            print("Génération " + str(generation))

    pygame.quit()
