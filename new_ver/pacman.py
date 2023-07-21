#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pygame
from pygame.locals import *
from new_ver.vector import Vectors
from new_ver.ghost import Ghost
#from constants import *
#pacman_col = (255, 255, 0)
pacman=0
stop = 0#STOP = 0
up = 1 #UP = 1 #
down =-1 #DOWN = -1
left = 2 #LEFT = 2
right =-2 #RIGHT = -2
portal =5
name1=0#PACMAN = 0
cellw = 16   #TILEWIDTH
cellh= 16
SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3
class mypacman(object): #Pacman
    def __init__(self,node):
        #Ghost.__init__(self, node )
        self.name = pacman 
        self.position = Vectors(200, 400)
        self.directions = {stop:Vectors(),left:Vectors(-1,0), right:Vectors(1,0), up:Vectors(0,-1), 
                           down:Vectors(0,1) }
        
        self.direction = right
        self.speed = 5
        self.radius = cellw/2 #can delete!
        #self.color = pacman_col
        self.node = node
        self.setPosition()
        self.startNode=node
        self.target = node
        self.alive = True
        
    def setStartNode(self, node):
        self.node = self.startNode
        #self.startNode = node
        self.target = node
        self.setPosition()
    def setBetweenNodes(self, direction):
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = (self.node.position + self.target.position) / 2.0

    def reset(self):
       # Ghost.reset(self)
        self.direction = right
        self.setBetweenNodes(left)
        self.alive = True

    def die(self):
        self.alive = False
        self.direction = stop

    def setPosition(self):
        self.position = self.node.position.copy()
    def key(self): #getValidKey
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_LEFT]:
            return left
        if key_pressed[K_RIGHT]:
            return right
        if key_pressed[K_UP]:
            return up
        if key_pressed[K_DOWN]:
            return down
        return stop
    
    def update(self,action=None):	
        self.position += self.directions[self.direction]*self.speed#*time #remove time?
        direction = self.key() if action is None else action
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[portal] is not None:
                self.node = self.node.neighbors[portal]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = stop
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.reverseDirection()
    def overshotTarget(self):
        if self.target is not None:
            vec1 = self.target.position - self.node.position
            vec2 = self.position - self.node.position
            node2Target = vec1.magnitudeSquared()
            node2Self = vec2.magnitudeSquared()
            return node2Self >= node2Target
        return False
    
    def validDirection(self, direction):
        if direction is not stop:
            if self.name in self.node.access[direction]:
                if self.node.neighbors[direction] is not None:
                    return True
        return False

    def getNewTarget(self, direction):
        if self.validDirection(direction):
            return self.node.neighbors[direction]
        return self.node
    
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
    def collideGhost(self, ghost): #reduce to just colide ?
        return self.collide(ghost)

    def collide(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.radius + other.radius)**2
        if dSquared <= rSquared:
            return True
        return False
    def eatDots(self, dots):
        for dot in dots:
            if self.collide(dot):
                return dot
        return None
    
    def draw(self, screen): #render
        p = self.position.asInt()
        pygame.draw.circle(screen, 'yellow', p, self.radius)

