cellw = 16   #TILEWIDTH
cellh= 16   #TILEHEIGHT
row = 36   #NROWS
col = 28  #NCOLS
w = col*cellw #SCREENWIDTH
h =row * cellh #SCREENHEIGHT = NROWS*TILEHEIGHT
screen = (w,h)
#SCREENSIZE = (SCREENWIDTH, SCREENHEIGHT)
#screen_color = (0, 0, 0)

#pacman_col = (255, 255, 0)

stop = 0#STOP = 0
up = 1 #UP = 1 #
down =-1 #DOWN = -1
left = 2 #LEFT = 2
right =-2 #RIGHT = -2

pacman=0#PACMAN = 0
SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3
powerdot=4

blinky = 4
pinky = 5
inky = 6
clyde = 7
import numpy as np
import pygame
from pygame.locals import *
from new_ver.pacman import mypacman 
from new_ver.nodes import NodeGroup,PelletGroup
from new_ver.ghost import GhostGroup
#from constants import * Everything above
class GameState:
    def __init__(self):
        self.lives = 0
        self.invalid_move = False
        self.total_pellets = 0
        self.collected_pellets = 0

class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(screen, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.lives = 3
        self.level = 0
        self.won=False
        self.lost=False
        self.score = 0
    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def end(self):
        self.pacman.visible = False
        self.ghosts.hide()
    def nextLevel(self):
        self.won=False
        self.lost=False
        self.level += 1
        #self.pause.paused = True
        self.showEntities()
        self.startGame()

    def restartGame(self):
        self.won=False
        self.lost=False
        self.lives = 3
        self.level = 0
        self.score = 0
        #self.pause.paused = True
        #self.fruit = None
        self.startGame()

    def resetLevel(self):
        #self.pause.paused = True
        self.pacman = mypacman(self.nodes.getNodeFromTiles(15, 26))
        self.ghosts.reset()
        #self.fruit = None
        
    def setBackground(self):
        self.background = pygame.surface.Surface(screen).convert()
        self.background.fill('black')
        
    def startGame(self):
        
        self.setBackground()
        self.nodes = NodeGroup("map.txt")
        self.nodes.setPortalPair((0,17), (27,17))# change numbers to letters
        box = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(box, (12,14), left)
        self.nodes.connectHomeNodes(box, (15,14), right)
        self.pellets = PelletGroup("map.txt")
        self.pacman = mypacman(self.nodes.getNodeFromTiles(15, 23))
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman )   
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2, 10))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.eatenPellets = []
        
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        
        self.nodes.denyHomeAccess(self.pacman)#!
        
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, left, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, right, self.ghosts)
        #self.ghosts.inky.startNode.denyAccess(right, self.ghosts.inky)
        #self.ghosts.clyde.startNode.denyAccess(left, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, up, self.ghosts)
        self.nodes.denyAccessList(15, 14, up, self.ghosts)
        self.nodes.denyAccessList(12, 26, up, self.ghosts)
        self.nodes.denyAccessList(15, 26, up, self.ghosts)

        
    def updateScore(self, points):
        self.score += points        
    def update(self): #remove time later !
        time = self.clock.tick(30) / 1000.0 #dt
        self.pacman.update(time)  #remove time?
        self.pellets.update(time)
        self.checkEvents()
        self.eatDots()
        self.checkGhostEvents()
        self.ghosts.update(time)
        self.render()
        self.get_frame()
    def get_frame(self):
        raw_maze_data = []
        with open('map.txt', 'r') as f:
            for line in f:
                raw_maze_data.append(line.split())
        raw_maze_data = np.array(raw_maze_data)
        self.state = np.zeros(raw_maze_data.shape)
        for idx, values in enumerate(raw_maze_data):
            for id, value in enumerate(values):
                if value in ['9', '=', 'X']:
                    self.state[idx][id] = 1
        # for idx, pellet in enumerate(self.eatenPellets):
        #     x = int(pellet.position.x / 16)
        #     y = int(pellet.position.y / 16)
        #     self.state[y][x] = 2
        for idx, pellet in enumerate(self.pellets.pelletList):
            x = int(pellet.position.x / 16)
            y = int(pellet.position.y / 16)
            if pellet.name == 3:
                self.state[y][x] = 3
            else:
                self.state[y][x] = 4
        x = int(round(self.pacman.position.x / 16))
        y = int(round(self.pacman.position.y / 16))
        self.state[y][x] = 5
        assert self.state[y][x] != 1
        for ghost in enumerate(self.ghosts):
            x = int(round(ghost[1].position.x / 16))
            y = int(round(ghost[1].position.y / 16))
            if ghost[1].mode.current is not FREIGHT and ghost[1].mode.current is not SPAWN:
                self.state[y][x] = -6
            else:
                self.state[y][x] = 6
        # dist = math.sqrt((self.pacman_prev.x - x)**2 + (self.pacman_prev.y - x)**2)
        # if abs(self.pacman_prev.x - x) >= 16 or abs(self.pacman_prev.y - y) >= 16:
        #     self.pacman_prev = self.pacman.position
        #     print("move",self.pacman.position)

        return self.state[3:34, :]
    def perform_action(self, action):
        state = None
        invalid_move = False
        if not self.pacman.validDirection(action):
            invalid_move = True
        time = self.clock.tick(60) / 1000.0 #dt
        self.pacman.update(time,ction=action)  #remove time?
        self.pellets.update(time)
        self.checkEvents()
        self.eatDots()
        self.checkGhostEvents()
        self.ghosts.update(time)
        self.render()
        state = self.get_frame()
        info = GameState()
        info.lives = self.lives
        info.invalid_move = invalid_move
        info.total_pellets = len(
            self.pellets.pelletList) + len(self.eatenPellets)
        info.collected_pellets = len(self.eatenPellets)
        return (state, self.score, self.lives == 0 or (self.pellets.isEmpty()), info)
    def eatDots(self):
        dot = self.pacman.eatDots(self.pellets.pelletList)
        if dot:
            self.pellets.numEaten += 1
            self.updateScore(dot.points)
            self.pellets.pelletList.remove(dot)
            #print("remain dots",len(self.pellets.pelletList))
            if dot.name == powerdot:
                   self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.end()
                self.won=True
                #self.nextLevel

    def checkGhostEvents(self):
        for ghost in self.ghosts:                        
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.nodes.allowHomeAccess(ghost)
                    ghost.startSpawn()    
                elif ghost.mode.current is not SPAWN:
                     if self.pacman.alive:
                            self.lives -=  1
                            self.pacman.die()
                            self.ghosts.hide()
                            if self.lives <= 0:
                                self.lost=True
                                #self.restartGame()
                            else:
                                 self.resetLevel()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
                pygame.quit()
            elif self.won and event.type == KEYDOWN:
                if event.key == K_SPACE:
                    self.restartGame()
                

    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.nodes.render(self.screen)
        self.pellets.render(self.screen)
        self.pacman.draw(self.screen)
        self.ghosts.render(self.screen)
        pygame.display.update()
        
        
        
if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()
