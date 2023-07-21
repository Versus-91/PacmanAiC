import pygame
from pygame.locals import *
from new_ver.vector import Vectors
#from constants import *
from random import randint
from new_ver.behavior import ModeController
stop = 0#STOP = 0
up = 1 #UP = 1 #
down =-1 #DOWN = -1
left = 2 #LEFT = 2
right =-2 #RIGHT = -2
cellw = 16   #TILEWIDTH
cellh= 16   #TILEHEIGHT
portal= 5
SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3
cellw = 16   #TILEWIDTH
cellh= 16   #TILEHEIGHT
row = 36   #NROWS
col = 28  #NCOLS
w = col*cellw #SCREENWIDTH
h =row * cellh #SCREENHEIGHT = NROWS*TILEHEIGHT
blinky = 4
pinky = 5
inky = 6
clyde = 7
class Ghost(object):
    def __init__(self, node,pacman=None,blinky=None):
        self.name = None
        self.directions = {stop:Vectors(),left:Vectors(-1,0), right:Vectors(1,0), up:Vectors(0,-1), 
                           down:Vectors(0,1) }
        self.direction = stop
        self.setSpeed(100)
        self.radius = 10
        self.collideRadius = 5
        self.color = 'blue'
        #self.node = node
        #self.setPosition()
        #self.target = node
        self.visible = True
        self.disablePortal = False
        self.goal = Vectors() #goal
        
        self.directionMethod = self.goalDirection
        
        
        self.pacman = pacman
        self.mode = ModeController(self)
        #!here before changes
        self.blinky = blinky
        self.homeNode = node
        #self.frozen=self.frozen()
        self.setStartNode(node)

    def setStartNode(self, node):
        self.node = node
        self.startNode = node
        self.target = node
        self.setPosition()
    def setBetweenNodes(self, direction):
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = (self.node.position + self.target.position) / 2.0

    def setPosition(self):
        self.position = self.node.position.copy()
          
    def validDirection1(self, direction):
        if direction is not stop:
            if self.name in self.node.access[direction]:
                if self.node.neighbors[direction] is not None:
                    return True
        return False

    def getNewTarget(self, direction):
        if self.validDirection1(direction):
            return self.node.neighbors[direction]
        return self.node

    def overshotTarget(self):
        if self.target is not None:
            vec1 = self.target.position - self.node.position
            vec2 = self.position - self.node.position
            node2Target = vec1.magnitudeSquared()
            node2Self = vec2.magnitudeSquared()
            return node2Self >= node2Target
        return False

    def reverseDirection(self):
        self.direction *= -1
        temp = self.node
        self.node = self.target
        self.target = temp
        
    def oppositeDirection(self, direction):
        if direction is not stop:
            if direction == self.direction * -1:
                return True
        return False

    def setSpeed(self, speed):
        self.speed = speed * cellw / 16
    def update(self,dt):
        self.position += self.directions[self.direction]*self.speed*dt
         
        if self.overshotTarget():
            self.node = self.target
            directions = self.validDirections()
            direction = self.randomDirection(directions) 
            #direction = self.directionMethod(directions)
            if not self.disablePortal:
                if self.node.neighbors[portal] is not None:
                    self.node = self.node.neighbors[portal]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            self.setPosition()
    def update_mode(self, dt):
        self.mode.update(dt)
        if self.mode.current is SCATTER:
            self.scatter()
        elif self.mode.current is CHASE:
            self.chase()
        self.update(dt)
    def frozen(self):
        self.mode.setFreightMode()
        if self.mode.current == FREIGHT:
            self.setSpeed(50)
            self.directionMethod = self.randomDirection  
            #self.color="green"

    def normalMode(self):
        self.setSpeed(100)
        self.directionMethod = self.goalDirection
        self.visible=True
        self.homeNode.denyAccess(down, self)
        #self.color="blue"
    #def scatter(self):
     #   self.goal = Vectors()

    #def chase(self):
    #    self.goal = self.pacman.position

    def spawn(self):
        self.goal = self.spawnNode.position

    def setSpawnNode(self, node):
        self.spawnNode = node

    def startSpawn(self):
        self.mode.setSpawnMode()
        if self.mode.current == SPAWN:
            self.setSpeed(150)
            self.directionMethod = self.goalDirection
            self.spawn()  
    def validDirections(self):
        directions = []
        for key in [up, down, left, right]:
            if self.validDirection1(key):
                if key != self.direction * -1:
                    directions.append(key)
        if len(directions) == 0:
            directions.append(self.direction * -1)
        return directions

    def randomDirection(self, directions):
        return directions[randint(0, len(directions)-1)]
    def goalDirection(self, directions):
        distances = []
        for direction in directions:
            vec = self.node.position  + self.directions[direction]*cellw - self.goal
            distances.append(vec.magnitudeSquared())
        index = distances.index(min(distances))
        return directions[index]
    def reset(self):
        self.setStartNode(self.startNode)
        self.direction = stop
        self.speed = 100
        self.visible = True  
        self.points = 200
        self.directionMethod = self.goalDirection
    def render(self, screen):
        if self.visible:
            p = self.position.asInt()
            pygame.draw.circle(screen, self.color, p, self.radius)
            
class Blinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name =clyde
        self.color = 'green'
    def scatter(self):
        self.goal = Vectors()

    def chase(self):
        self.goal = self.pacman.position
class Clyde(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = clyde
        self.color = 'orange'

    def scatter(self):
        self.goal = Vectors(0, h)

    def chase(self):
        d = self.pacman.position - self.position
        ds = d.magnitudeSquared()
        if ds <= (cellw * 5)**2:
            self.scatter()
        else:
            self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * cellw * 5
        
class Pinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = pinky
        self.color = 'pink'

    def scatter(self):
        self.goal = Vectors(w, 0)

    def chase(self):
        self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * cellw * 5
class Inky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = inky
        self.color = 'blue'

    def scatter(self):
        self.goal = Vectors(w, h) # 

    def chase(self):
        vec1 = self.pacman.position + self.pacman.directions[self.pacman.direction] * cellw * 2
        vec2 = (vec1 - self.blinky.position) * 2
        self.goal = self.blinky.position + vec2

class GhostGroup(object):
    def __init__(self, node, pacman):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        self.inky = Inky(node, pacman, self.blinky)
        self.clyde = Clyde(node, pacman)
        self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]

    def __iter__(self):
        return iter(self.ghosts)

    def update(self, dt):
        for ghost in self:
            ghost.update_mode(dt)

    def startFreight(self):
        for ghost in self:
            ghost.frozen()
        self.resetPoints()

    def setSpawnNode(self, node):
        for ghost in self:
            ghost.setSpawnNode(node)

    def updatePoints(self):
        for ghost in self:
            ghost.points *= 2

    def resetPoints(self):
        for ghost in self:
            ghost.points = 200

    def reset(self):
        for ghost in self:
            ghost.reset()

    def hide(self):
        for ghost in self:
            ghost.visible = False

    def show(self):
        for ghost in self:
            ghost.visible = True

    def render(self, screen):
        for ghost in self:
            ghost.render(screen)
