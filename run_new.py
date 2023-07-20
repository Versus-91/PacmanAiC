cellw = 16   #TILEWIDTH
cellh= 16   #TILEHEIGHT
row = 36   #NROWS
col = 28  #NCOLS
w = col*cellw #SCREENWIDTH
h =row * cellh #SCREENHEIGHT = NROWS*TILEHEIGHT
screen = (w,h)
#SCREENSIZE = (SCREENWIDTH, SCREENHEIGHT)

stop = 0
up = 1 
down =-1 
left = 2 
right =-2 

pacman=0
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
from bfs import minDistance
from new_ver.pacman import mypacman 
from new_ver.nodes import NodeGroup,PelletGroup
from new_ver.ghost import GhostGroup
player_images =[]
for i in range (1,5):
    player_images.append(pygame.transform.scale(pygame.image.load(f'assets/player_images/{i}.png'),(cellw/2,cellw/2)))

class GameState:
    def __init__(self):
        self.lives = 0
        self.frame = []
        self.invalid_move = False
        self.total_pellets = 0
        self.collected_pellets = 0
        self.food_distance = -1
        self.powerup_distance =-1
        self.ghost_distance = -1
        self.scared_ghost_distance = -1
        self.x = 0
        self.y = 0

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
        
        self.timer=0
        self.font=pygame.font.Font('freesansbold.ttf', 20)
        
        self.counter=0
        self.level_map="map.txt"
        
    def draw_misc(self):
        score_text = self.font.render(f'Score:{self.score}',True,'white')
        self.screen.blit(score_text,(10,h-20))
        if self.ghosts.pinky.mode.current == FREIGHT or self.ghosts.inky.mode.current == FREIGHT or self.ghosts.clyde.mode.current == FREIGHT or self.ghosts.blinky.mode.current == FREIGHT:
            
            pygame.draw.circle(self.screen,'yellow',(140,h-15),cellw*2/3)   #powerup is active
            #print("powerup")
        for i in range(self.lives):
            self.screen.blit(pygame.transform.scale(player_images[0],(cellw,cellh)),(w-180+i*40,h-20))
        if self.lost:
            pygame.draw.rect(self.screen,'white',[w/4-20,h/4,w/2+25,h/2],0,10 )
            pygame.draw.rect(self.screen,'black',[w/4-5,h/4+15,w/2,h/2-30],0,10 )
            menu_text=self.font.render('You lost!',True,'red')
            self.screen.blit(menu_text,(w/4,h/2))
            #print("lost")
        if self.won:
            pygame.draw.rect(self.screen,'white',[w/4-20,h/4,w/2+25,h/2],0,10 )
            pygame.draw.rect(self.screen,'black',[w/4-5,h/4+15,w/2,h/2-30],0,10 )
            menu_text=self.font.render('You won!',True,'green')
            self.screen.blit(menu_text,(w/4,h/2))
            #print("won")
            
    
    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def end(self):
        self.pacman.visible = False
        self.ghosts.hide()


    def restartGame(self):
        self.won=False
        self.lost=False
        self.lives = 3
        self.level = 0
        self.score = 0
        
        self.startGame()
    def nextLevel(self):
        timer=0
        self.pacman.visible = False
        self.ghosts.hide()
        
        self.level+=1
        self.won=False
        self.level_map="maze1.txt"
        self.nodes = NodeGroup(self.level_map)
        self.nodes.setPortalPair((0,17), (27,17))# change numbers to letters
        box = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(box, (12,14), left)
        self.nodes.connectHomeNodes(box, (15,14), right)
        self.pellets = PelletGroup(self.level_map)
        
        self.eatenPellets = []
        self.nodes.default_color='red'
        
        self.pacman = mypacman(self.nodes.getNodeFromTiles(15, 23))
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2, 8))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
    def resetLevel(self):
       
        self.pacman = mypacman(self.nodes.getNodeFromTiles(15, 26))
        self.ghosts.reset()
        
        
    def setBackground(self):
        self.background = pygame.surface.Surface(screen).convert()
        self.background.fill('black')
        
    def startGame(self):
        
        self.setBackground()
        self.nodes = NodeGroup(self.level_map)
        self.nodes.setPortalPair((0,17), (27,17))# change numbers to letters
        box = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(box, (12,14), left)
        self.nodes.connectHomeNodes(box, (15,14), right)
        self.pellets = PelletGroup(self.level_map)
        self.pacman = mypacman(self.nodes.getNodeFromTiles(15, 26))
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman )   
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(1, 4))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.eatenPellets = []
        
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        
        self.nodes.denyHomeAccess(self.pacman)#!
        
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, left, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, right, self.ghosts)
        
        self.nodes.denyAccessList(12, 14, up, self.ghosts)
        self.nodes.denyAccessList(15, 14, up, self.ghosts)
        self.nodes.denyAccessList(12, 26, up, self.ghosts)
        self.nodes.denyAccessList(15, 26, up, self.ghosts)
        #self.draw_misc()
        
    def updateScore(self, points):
        self.score += points        
    def update(self): #remove time later !
        time = self.clock.tick(60) / 1000.0 
        self.pellets.update(time)
        self.checkGhostEvents()
        self.checkEvents()
        self.eatDots()
        self.pacman.update(time)  #remove time?
        self.ghosts.blinky.pacman = self.pacman
        self.ghosts.update(time)
        self.render()
        self.get_frame()
        if self.counter < 19: #spped of eating my pacman 
            self.counter += 1
        else:
            self.counter=0
        total_pellets = len(
        self.pellets.pelletList) + len(self.eatenPellets)
        collected_pellets = len(self.eatenPellets)
    def get_frame(self):
        raw_maze_data = []
        with open('map.txt', 'r') as f:
            for line in f:
                raw_maze_data.append(line.split())
        raw_maze_data = np.array(raw_maze_data)
        self.state = np.zeros(raw_maze_data.shape)
        for idx, values in enumerate(raw_maze_data):
            for id, value in enumerate(values):
                if value in ['9', '=', 'X','3','4','5','6','7','8']:
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
        pacman_x = int(round(self.pacman.position.x / 16))
        pacman_y = int(round(self.pacman.position.y / 16))
        self.state[pacman_y][pacman_x] = 5
        assert self.state[y][x] != 1
        for ghost in enumerate(self.ghosts):
            x = int(round(ghost[1].position.x / 16))
            y = int(round(ghost[1].position.y / 16))
            if ghost[1].mode.current is not FREIGHT and ghost[1].mode.current is not SPAWN:
                self.state[y][x] = -6
            elif ghost[1].mode.current is FREIGHT:
                if self.state[y][x] != 5:
                    self.state[y][x] = 6
        # dist = math.sqrt((self.pacman_prev.x - x)**2 + (self.pacman_prev.y - x)**2)
        # if abs(self.pacman_prev.x - x) >= 16 or abs(self.pacman_prev.y - y) >= 16:
        #     self.pacman_prev = self.pacman.position
        #     print("move",self.pacman.position)

        return self.state[3:34, :]
    def perform_action(self, action):
        info = GameState()
        if self.lost == True:
            self.lost == False
        info.frame = self.get_frame()
        invalid_move = False
        lives = self.lives
        if not self.pacman.validDirection(action):
            invalid_move = True
        time = self.clock.tick(60) / 1000.0 #dt

        self.pellets.update(time)
        self.checkEvents()
        self.eatDots()
        self.checkGhostEvents()
        self.pacman.update(time,action=action)  #remove time?
        self.ghosts.blinky.pacman = self.pacman
        self.ghosts.update(time)

        self.render()
        if lives == self.lives:
            info.frame = self.get_frame()
        done = self.lost
        row_indices, _ = np.where(info.frame == 5)
        info.invalid_move = invalid_move
        info.total_pellets = len(
        self.pellets.pelletList) + len(self.eatenPellets)
        info.collected_pellets = len(self.eatenPellets)
        info.lives = self.lives

        if row_indices.size > 0:
            info.food_distance = minDistance(info.frame,5,3,[-6,1])
            info.powerup_distance = minDistance(info.frame,5,4,[-6,1])
            info.ghost_distance = minDistance(info.frame,5,-6)
            info.scared_ghost_distance = minDistance(info.frame,5,6)
        return ([], self.score, done, info)
    def eatDots(self):
        dot = self.pacman.eatDots(self.pellets.pelletList)
        if dot:
            self.eatenPellets.append(dot)
            self.pellets.numEaten += 1
            self.updateScore(dot.points)
            self.pellets.pelletList.remove(dot)
            #print("remain dots",len(self.pellets.pelletList))
            if len(self.pellets.pelletList) < 15:
                self.ghosts.pinky.gets_angry(self.counter)
            if dot.name == powerdot:
                self.ghosts.startFreight()
                
            if self.pellets.isEmpty():
                self.end()
                self.won=True
                #self.nextLevel()

    def checkGhostEvents(self):
        for ghost in self.ghosts:                        
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    # ghost.visible = False
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
        self.nodes.render(self.screen,self.level_map)
        self.pellets.render(self.screen)
        self.pacman.draw(self.screen,self.counter)
        self.ghosts.render(self.screen,self.counter)
        self.draw_misc()
        pygame.display.update()
        
        
        
if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()