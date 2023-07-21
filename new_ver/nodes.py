import numpy as np
import pygame
from new_ver.vector import Vectors
import math
#from constants import *
WHITE = (255, 255, 255)
RED = (255, 0, 0)
stop = 0#STOP = 0
up = 1 #UP = 1 #      3
down =-1 #DOWN = -1         4
left = 2 #LEFT = 2        1
right =-2 #RIGHT = -2       2
portal= 5
dot=3
powerdot=4
pacman=0
blinky = 4
pinky = 5
inky = 6
clyde = 7
sue=8
cellw = 16   #TILEWIDTH
cellh= 16   #TILEHEIGHT
class Node(object):
    def __init__(self, x, y):
        self.position = Vectors(x, y)
        self.neighbors = {up:None, down:None, left:None, right:None,portal:None}
        self.access = {up:[pacman, blinky, pinky, inky, clyde],
                       down:[pacman, blinky, pinky, inky, clyde],
                       left:[pacman, blinky, pinky, inky, clyde],
                       right:[pacman,blinky, pinky, inky, clyde],}
      

    def denyAccess(self, direction, ghost):
        if ghost.name in self.access[direction]:
            self.access[direction].remove(ghost.name)

    def allowAccess(self, direction, ghost):
        if ghost.name not in self.access[direction]:
            self.access[direction].append(ghost.name)
            
    def draw(self, screen):
        for n in self.neighbors.keys():
            if self.neighbors[n] is not None:
                line_start = self.position.asTuple()
                line_end = self.neighbors[n].position.asTuple()
                pygame.draw.line(screen, "white", line_start, line_end, 4)
                pygame.draw.circle(screen, "red", self.position.asInt(), 12)

class NodeGroup(object):
    def __init__(self,level):
        #self.nodeList = []
        self.level = level
        self.cell = {}
        self.allnode = ['1','2','n']   # we assign 1-to node element 
        self.path = ['0','-']   #0 to path element
        self.box = None
        data = self.read_map(level)
        self.createNodeTable(data)
        self.connectHorizontally(data)
        self.connectVertically(data)
        self.default_color='blue'
        self.pi=math.pi

    def read_map(self, textfile):#readMazeFile
        return np.loadtxt(textfile, dtype='<U1')
    
    def createNodeTable(self, data,x1=0, y1=0):  
        for i in list(range(data.shape[0])):
            for j in list(range(data.shape[1])):
                if data[i][j] in self.allnode:
                    x, y = self.getKey(j+x1, i+y1)
                    self.cell[(x, y)] = Node(x, y)
                    
    def createHomeNodes(self, x, y):
        homedata = np.array([['X','X','1','X','X'],
                             ['X','X','0','X','X'],
                             ['1','X','0','X','1'],
                             ['1','0','1','0','1'],
                             ['1','X','X','X','1']])

        self.createNodeTable(homedata, x, y)
        self.connectHorizontally(homedata, x, y)
        self.connectVertically(homedata, x, y)
        self.box = self.getKey(x+2, y)
        return self.box
    def connectHomeNodes(self, homekey, otherkey, direction):     
        key = self.getKey(*otherkey)
        self.cell[homekey].neighbors[direction] = self.cell[key]
        self.cell[key].neighbors[direction*-1] = self.cell[homekey]

    def getKey(self, x, y):
        return x * cellw, y * cellh
    
    def getNodeFromPixels(self, x, y):
        if (x, y) in self.cell.keys():
            return self.cell[(x, y)]
        return None

    def getNodeFromTiles(self, col, row):
        x, y = self.getKey(col, row)
        if (x, y) in self.cell.keys():
            return self.cell[(x, y)]
        
   
    def getStartTempNode(self):
        nodes = list(self.cell.values())
        return nodes[1]
    def connectVertically(self, data,x=0, y=0):
        dataT = data.transpose()
        for i in list(range(dataT.shape[0])):
            key = None
            for j in list(range(dataT.shape[1])):
                if dataT[i][j] in self.allnode:
                    if key is None:
                        key = self.getKey(i+x, j+y)
                    else:
                        otherkey = self.getKey(i+x, j+y)
                        self.cell[key].neighbors[down] = self.cell[otherkey]
                        self.cell[otherkey].neighbors[up] = self.cell[key]
                        key = otherkey
                elif dataT[i][j] not in self.path:
                    key = None
    def connectHorizontally(self, data,x=0, y=0):
        
        for i in list(range(data.shape[0])):
            key = None
            for j in list(range(data.shape[1])):
                if data[i][j] in self.allnode:
                    if key is None:
                        key = self.getKey(j+x, i+y)
                    else:
                        otherkey = self.getKey(j+x, i+y)
                        self.cell[key].neighbors[right] = self.cell[otherkey]
                        self.cell[otherkey].neighbors[left] = self.cell[key]
                        key = otherkey
                elif data[i][j] not in self.path:
                    key = None

                    
    def denyAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.denyAccess(direction, entity)

    def allowAccess(self, col, row, direction, entity):
        node = self.getNodeFromTiles(col, row)
        if node is not None:
            node.allowAccess(direction, entity)

    def denyAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.denyAccess(col, row, direction, entity)

    def allowAccessList(self, col, row, direction, entities):
        for entity in entities:
            self.allowAccess(col, row, direction, entity)

    def denyHomeAccess(self, entity):
        self.cell[self.box].denyAccess(down, entity)

    def allowHomeAccess(self, entity):
        self.cell[self.box].allowAccess(down, entity)

    def denyHomeAccessList(self, entities):
        for entity in entities:
            self.denyHomeAccess(entity)

    def allowHomeAccessList(self, entities):
        for entity in entities:
            self.allowHomeAccess(entity) 
            
            
            
            
            # 0 = empty black rectangle, 1 = dot, 2 = big dot, 3 = vertical line,
# 4 = horizontal line, 5 = top right, 6 = top left, 7 = bot left, 8 = bot right
# 9 = gate
 
        
    def render(self, screen,level=None):
        data=self.read_map(level)
        for i in list(range(data.shape[0])):
            for j in list(range(data.shape[1])):
                

                if data[i][j] == '3':
                    
                    pygame.draw.line(screen, self.default_color, (j * cellw ,
                                                                           i * cellh-0.5*cellh), (j * cellw , i * cellh + cellh-0.5*cellh), 3)
                    
                if data[i][j] == '4':
                    pygame.draw.line(screen, self.default_color, (j * cellw- 0.5*cellw, i * cellh ), (j * cellw + 0.5*cellw, i * cellh + 0.5 *cellh-0.5*cellh), 3)

                if data[i][j] == '5':
                    pygame.draw.arc(screen, self.default_color, [(j * cellw - cellw), i * cellh , cellw, cellh], 0, self.pi/2, 3)    
                if data[i][j] == '6':
                    pygame.draw.arc(screen, self.default_color, [(j * cellw ), i * cellh , cellw, cellh], self.pi/2, self.pi, 3)
                if data[i][j] == '7':
                    pygame.draw.arc(screen, self.default_color, [(j * cellw ), i * cellh - cellh, cellw, cellh], self.pi, 3*self.pi/2, 3)
                if data[i][j] == '8':
                    pygame.draw.arc(screen, self.default_color, [(j * cellw - cellw), i *cellh - cellh, cellw, cellh], 3*self.pi/2, self.pi*2, 3)

                if data[i][j] == '9':
                    pygame.draw.line(screen, 'white', (j * cellw- 0.5*cellw, i * cellh ), (j * cellw + 0.5*cellw, i * cellh + 0.5 *cellh-0.5*cellh), 3)

        

                
                
     #   for i in self.cell.values():
      #      if i.neighbors[portal] ==  None:  # here ! we dont draw portal node
       #         i.draw(screen)
                
       
    def setPortalPair(self, pair1, pair2):
        key1 = self.getKey(*pair1)
        key2 = self.getKey(*pair2)
        if key1 in self.cell.keys() and key2 in self.cell.keys():
            self.cell[key1].neighbors[portal] = self.cell[key2]
            self.cell[key2].neighbors[portal] = self.cell[key1]

        return None      
            
class Dots(object):
    def __init__(self, row, column):
        self.name = dot
        self.position = Vectors(column*cellw, row*cellh)
        self.color = WHITE
        self.radius = int(4 * cellw / 16)
        self.collideRadius = int(4 * cellh / 16)
        self.points = 10
        self.visible = True
        
    def render(self, screen):
        if self.visible:
            p = self.position.asInt()
            pygame.draw.circle(screen, 'white', p, self.radius)

class Powerdots(Dots):
    def __init__(self, row, column):
        Dots.__init__(self, row, column)
        self.name = powerdot
        self.radius = int(8 * cellw / 16)
        self.points = 50
        self.flashTime = 0.2
        self.timer= 0
        
    def update(self, dt):
        self.timer += dt
        # if self.timer >= self.flashTime:
        #     self.visible = not self.visible
        #     self.timer = 0
            
class PelletGroup(object):
    def __init__(self, pelletfile):
        self.pelletList = []
        self.powerpellets = []
        self.createPelletList(pelletfile)
        self.numEaten = 0

    def update(self, dt):
        for powerpellet in self.powerpellets:
            powerpellet.update(dt)
                
    def createPelletList(self, pelletfile):
        data = self.readPelletfile(pelletfile)        
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in ['0', '1']:
                    self.pelletList.append(Dots(row, col))
                elif data[row][col] in ['2']:
                    pp = Powerdots(row, col)
                    self.pelletList.append(pp)
                    self.powerpellets.append(pp)
                    
    def readPelletfile(self, textfile):
        return np.loadtxt(textfile, dtype='<U1')
    
    def isEmpty(self):
        if len(self.pelletList) == 0:
            return True
        return False
    
    def render(self, screen):
        for pellet in self.pelletList:
            pellet.render(screen)