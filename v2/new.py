import pygame
from pygame.locals import *
import math
import numpy as np
from random import randint



############################################## Constants
cell_side_size = 16
cell_side_midlle = cell_side_size / 2
n_rows = 36
n_columns = 28
window_width = n_columns*cell_side_size
window_height = n_rows*cell_side_size
window_frame_size = (window_width, window_height)

stop_indicator = 0
up_direction = 1
down_direction = -1
left_direction = 2
right_direction = -2
in_portal = 3

pacman_indicator = 0
objectives_indicator = 1
powerups_indicator = 2
ghosts_indicator = 3
blinky_indicator = 4
pinky_indicator = 5
inky_indicator = 6
clyde_indicator = 7
gold_indicator = 8

scattering_mode = 0
chasing_mode = 1
scared_mode = 2
respawning_mode = 3

pacman_icon=pygame.transform.scale(pygame.image.load(f'assets/Pacman.png'),(2*cell_side_size,2*cell_side_size))
blinky_icon=pygame.transform.scale(pygame.image.load(f'assets/Blinky.png'),(2*cell_side_size,2*cell_side_size))
inky_icon=pygame.transform.scale(pygame.image.load(f'assets/Inky.png'),(2*cell_side_size,2*cell_side_size))
pinky_icon=pygame.transform.scale(pygame.image.load(f'assets/Pinky.png'),(2*cell_side_size,2*cell_side_size))
clyde_icon=pygame.transform.scale(pygame.image.load(f'assets/Clyde.png'),(2*cell_side_size,2*cell_side_size))
dead_icon=pygame.transform.scale(pygame.image.load(f'assets/Dead.png'),(2*cell_side_size,2*cell_side_size))
scared_icon=pygame.transform.scale(pygame.image.load(f'assets/Scared.png'),(2*cell_side_size,2*cell_side_size))
gold_icon=pygame.transform.scale(pygame.image.load(f'assets/Gold.png'),(2*cell_side_size,2*cell_side_size))
heart_icon=pygame.transform.scale(pygame.image.load(f'assets/Heart.png'),(2*cell_side_size,2*cell_side_size))

class GameState:
    def __init__(self):
        self.lives = 0
        self.invalid_move = False
        self.total_pellets = 0
        self.collected_pellets = 0

############################################## Vector
class CustomVector(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.epsilon = 0.0001

    def __str__(self):
        return "<"+str(self.x)+", "+str(self.y)+">"
    
    def __add__(self, other):
        return CustomVector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return CustomVector(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return CustomVector(-self.x, -self.y)

    def __mul__(self, scalar):
        return CustomVector(self.x * scalar, self.y * scalar)

    def __div__(self, scalar):
        if scalar != 0:
            return CustomVector(self.x / float(scalar), self.y / float(scalar))
        return None

    def __truediv__(self, scalar):
        return self.__div__(scalar)

    def __eq__(self, other):
        if abs(self.x - other.x) < self.epsilon:
            if abs(self.y - other.y) < self.epsilon:
                return True
        return False

    def magnitudeSquared(self):
        return self.x**2 + self.y**2

    def tupleForm(self):
        return self.x, self.y

    def intForm(self):
        return int(self.x), int(self.y)
    
    def copyVector(self):
        return CustomVector(self.x, self.y)



############################################## Nodes
class Node(object):
    def __init__(self, x, y):
        self.position = CustomVector(x, y)
        self.neighbors = {up_direction:None, down_direction:None, left_direction:None, right_direction:None, in_portal:None}
        self.possible_path = {up_direction:[pacman_indicator, blinky_indicator, pinky_indicator, inky_indicator, clyde_indicator, gold_indicator], 
                       down_direction:[pacman_indicator, blinky_indicator, pinky_indicator, inky_indicator, clyde_indicator, gold_indicator], 
                       left_direction:[pacman_indicator, blinky_indicator, pinky_indicator, inky_indicator, clyde_indicator, gold_indicator], 
                       right_direction:[pacman_indicator, blinky_indicator, pinky_indicator, inky_indicator, clyde_indicator, gold_indicator]}

    def removePath(self, direction, single_object):
        if single_object.name in self.possible_path[direction]:
            self.possible_path[direction].remove(single_object.name)

    def addPath(self, direction, single_object):
        if single_object.name not in self.possible_path[direction]:
            self.possible_path[direction].append(single_object.name)

############################################## Node Group
class NodeGroup(object):
    def __init__(self, level):
        self.level = level
        self.nodes_list = {}
        self.node_indicator = ['+', 'P', 'n']
        self.path_indicator = ['.', '-', '|', 'p']
        map = self.loadMaze(level)
        self.nodeTable(map)
        self.horizontalLines(map)
        self.verticalLines(map)
        self.box_key = None

    def loadMaze(self, level):
        return np.loadtxt(level, dtype='<U1')

    def nodeTable(self, map, x_value=0, y_value=0):
        for row in list(range(map.shape[0])):
            for col in list(range(map.shape[1])):
                if map[row][col] in self.node_indicator:
                    x, y = self.findKey(col+x_value, row+y_value)
                    self.nodes_list[(x, y)] = Node(x, y)

    def findKey(self, x, y):
        return x * cell_side_size, y * cell_side_size

    def horizontalLines(self, map, x_value=0, y_value=0):
        for row in list(range(map.shape[0])):
            key = None
            for col in list(range(map.shape[1])):
                if map[row][col] in self.node_indicator:
                    if key is None:
                        key = self.findKey(col+x_value, row+y_value)
                    else:
                        key_prime = self.findKey(col+x_value, row+y_value)
                        self.nodes_list[key].neighbors[right_direction] = self.nodes_list[key_prime]
                        self.nodes_list[key_prime].neighbors[left_direction] = self.nodes_list[key]
                        key = key_prime
                elif map[row][col] not in self.path_indicator:
                    key = None

    def verticalLines(self, map, x_value=0, y_value=0):
        dataT = map.transpose()
        for col in list(range(dataT.shape[0])):
            key = None
            for row in list(range(dataT.shape[1])):
                if dataT[col][row] in self.node_indicator:
                    if key is None:
                        key = self.findKey(col+x_value, row+y_value)
                    else:
                        key_prime = self.findKey(col+x_value, row+y_value)
                        self.nodes_list[key].neighbors[down_direction] = self.nodes_list[key_prime]
                        self.nodes_list[key_prime].neighbors[up_direction] = self.nodes_list[key]
                        key = key_prime
                elif dataT[col][row] not in self.path_indicator:
                    key = None


    def startingNode(self):
        nodes = list(self.nodes_list.values())
        return nodes[0]

    def conncetPortalCells(self, cell_1, cell_2):
        pixel_1 = self.findKey(*cell_1)
        pixel_2 = self.findKey(*cell_2)
        if pixel_1 in self.nodes_list.keys() and pixel_2 in self.nodes_list.keys():
            self.nodes_list[pixel_1].neighbors[in_portal] = self.nodes_list[pixel_2]
            self.nodes_list[pixel_2].neighbors[in_portal] = self.nodes_list[pixel_1]

    def createBox(self, x_value, y_value):
        box_data = np.array([['X','X','+','X','X'],
                             ['X','X','.','X','X'],
                             ['+','X','.','X','+'],
                             ['+','.','+','.','+'],
                             ['+','X','X','X','+']])

        self.nodeTable(box_data, x_value, y_value)
        self.horizontalLines(box_data, x_value, y_value)
        self.verticalLines(box_data, x_value, y_value)
        self.box_key = self.findKey(x_value+2, y_value)
        return self.box_key

    def connectBoxNodes(self, box_key, key_prime, direction):     
        key = self.findKey(*key_prime)
        self.nodes_list[box_key].neighbors[direction] = self.nodes_list[key]
        self.nodes_list[key].neighbors[direction*-1] = self.nodes_list[box_key]

    def nodeFromCells(self, col, row):
        x, y = self.findKey(col, row)
        if (x, y) in self.nodes_list.keys():
            return self.nodes_list[(x, y)]
        return None

    def removePath(self, col, row, direction, single_object):
        node = self.nodeFromCells(col, row)
        if node is not None:
            node.removePath(direction, single_object)

    def addPath(self, col, row, direction, single_object):
        node = self.nodeFromCells(col, row)
        if node is not None:
            node.addPath(direction, single_object)

    def blockedPathList(self, col, row, direction, entities):
        for single_object in entities:
            self.removePath(col, row, direction, single_object)



############################################## Objects
class MovingObject(object):
    def __init__(self, node):
        self.name = None
        self.directions = {up_direction:CustomVector(0, -1),down_direction:CustomVector(0, 1), 
                          left_direction:CustomVector(-1, 0), right_direction:CustomVector(1, 0), stop_indicator:CustomVector()}
        self.direction = stop_indicator
        self.setSpeed(100)
        self.radius = 10
        self.collision_distance = 5
        self.color = None
        self.visible = True
        self.disablePortal = False
        self.goal = None
        self.moving_method = None
        self.initialNode(node)
        self.image = None

    def setPosition(self):
        self.position = self.node.position.copyVector()

    def update(self, delta_t):
        self.position += self.directions[self.direction]*self.speed*delta_t
         
        if self.passedTarget():
            self.node = self.target
            directions = self.allowedDirectionsList()
            direction = self.moving_method(directions)
            if not self.disablePortal:
                if self.node.neighbors[in_portal] is not None:
                    self.node = self.node.neighbors[in_portal]
            self.target = self.setTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.setTarget(self.direction)

            self.setPosition()
          
    def allowedDirection(self, direction):
        if direction is not stop_indicator:
            if self.name in self.node.possible_path[direction]:
                if self.node.neighbors[direction] is not None:
                    return True
        return False

    def setTarget(self, direction):
        if self.allowedDirection(direction):
            return self.node.neighbors[direction]
        return self.node

    def passedTarget(self):
        if self.target is not None:
            vector_1 = self.target.position - self.node.position
            vector_2 = self.position - self.node.position
            length_1 = vector_1.magnitudeSquared()
            length_2 = vector_2.magnitudeSquared()
            return length_2 >= length_1
        return False

    def turnMove(self):
        self.direction *= -1
        temp_value = self.node
        self.node = self.target
        self.target = temp_value
        
    def oppositeDirection(self, direction):
        if direction is not stop_indicator:
            if direction == self.direction * -1:
                return True
        return False

    def allowedDirectionsList(self):
        directions = []
        for key in [up_direction, down_direction, left_direction, right_direction]:
            if self.allowedDirection(key):
                if key != self.direction * -1:
                    directions.append(key)
        if len(directions) == 0:
            directions.append(self.direction * -1)
        return directions

    def randomMove(self, directions):
        return directions[randint(0, len(directions)-1)]

    def toTargetDirection(self, directions):
        distances = []
        for direction in directions:
            vec = self.node.position  + self.directions[direction]*cell_side_size - self.goal
            distances.append(vec.magnitudeSquared())
        index = distances.index(min(distances))
        return directions[index]

    def initialNode(self, node):
        self.node = node
        self.startNode = node
        self.target = node
        self.setPosition()

    def middleOfNodes(self, direction):
        if self.node.neighbors[direction] is not None:
            self.target = self.node.neighbors[direction]
            self.position = (self.node.position + self.target.position) / 2.0

    def reset(self):
        self.initialNode(self.startNode)
        self.direction = stop_indicator
        self.speed = 100
        self.visible = True

    def setSpeed(self, speed):
        self.speed = speed

    def render(self, screen):
        if self.visible:
            adjust_position = self.position - CustomVector(cell_side_midlle, cell_side_midlle)
            screen.blit(self.image, adjust_position.tupleForm())




############################################## Pacman
class Pacman(MovingObject):
    def __init__(self, node):
        MovingObject.__init__(self, node )
        self.name = pacman_indicator    
        self.color = 'yellow'
        self.direction = left_direction
        self.middleOfNodes(left_direction)
        self.alive = True
        self.image = pacman_icon

    def reset(self):
        MovingObject.reset(self)
        self.direction = left_direction
        self.middleOfNodes(left_direction)
        self.alive = True

    def die(self):
        self.alive = False
        self.direction = stop_indicator

    def update(self, delta_t,action = None):
        self.position += self.directions[self.direction]*self.speed*delta_t
        direction = self.getInput() if action is None else action
        if self.passedTarget():
            self.node = self.target
            if self.node.neighbors[in_portal] is not None:
                self.node = self.node.neighbors[in_portal]
            self.target = self.setTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.setTarget(self.direction)

            if self.target is self.node:
                self.direction = stop_indicator
            self.setPosition()
        else: 
            if self.oppositeDirection(direction):
                self.turnMove()

    def getInput(self):
        key_pressed = pygame.key.get_pressed()
        if key_pressed[K_UP]:
            return up_direction
        if key_pressed[K_DOWN]:
            return down_direction
        if key_pressed[K_LEFT]:
            return left_direction
        if key_pressed[K_RIGHT]:
            return right_direction
        return stop_indicator
    
    def collideCheck(self, other_unit):
        distance_squared = (self.position - other_unit.position).magnitudeSquared()
        colider_squared = (self.collision_distance + other_unit.collision_distance)**2
        if distance_squared <= colider_squared:
            return True
        return False

    def collectObjectives(self, objectivesList):
        for objective in objectivesList:
            if self.collideCheck(objective):
                return objective
        return None
    
    def collideGhost(self, ghost):
        return self.collideCheck(ghost)



############################################## Objective
class Objective(object):
    def __init__(self, row, column):
        self.name = objectives_indicator
        self.position = CustomVector(column*cell_side_size, row*cell_side_size)
        self.color = 'white'
        self.radius = 3
        self.collision_distance = 2
        self.point = 10
        self.visible = True
        
    def render(self, screen):
        if self.visible:
            adjust_position = self.position + CustomVector(cell_side_midlle, cell_side_midlle)
            pygame.draw.circle(screen, self.color, adjust_position.intForm(), self.radius)

############################################## Powerups
class PowerUp(Objective):
    def __init__(self, row, column):
        Objective.__init__(self, row, column)
        self.name = powerups_indicator
        self.radius = 8
        self.point = 50
        self.timer= 0
        
    def update(self, delta_t):
        self.timer += delta_t

############################################## Objectives Group
class ObjectiveGroup(object):
    def __init__(self, objective_data):
        self.objectivesList = []
        self.powerups = []
        self.createObjectivesList(objective_data)
        self.n_reached_objectives = 0

    def update(self, delta_t):
        for powerup in self.powerups:
            powerup.update(delta_t)
                
    def createObjectivesList(self, objective_data):
        data = self.loadObjectiveData(objective_data)        
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in ['.', '+']:
                    self.objectivesList.append(Objective(row, col))
                elif data[row][col] in ['P', 'p']:
                    pu = PowerUp(row, col)
                    self.objectivesList.append(pu)
                    self.powerups.append(pu)
                    
    def loadObjectiveData(self, level):
        return np.loadtxt(level, dtype='<U1')
    
    def noObjective(self):
        if len(self.objectivesList) == 0:
            return True
        return False
    
    def render(self, screen):
        for objective in self.objectivesList:
            objective.render(screen)



############################################## Ghosts
class Ghost(MovingObject):
    def __init__(self, node, pacman=None, blinky=None):
        MovingObject.__init__(self, node)
        self.name = ghosts_indicator
        self.point = 200
        self.goal = CustomVector()
        self.moving_method = self.toTargetDirection
        self.pacman = pacman
        self.mode = SwitchMode(self)
        self.blinky = blinky
        self.homeNode = node
        self.image = None
        #self.default_image = None
        #self.scared_image = scared_icon
        #self.dead_image = dead_icon
        #self.image = self.default_image

    def reset(self):
        MovingObject.reset(self)
        self.point = 200
        self.moving_method = self.toTargetDirection
        #self.image = self.default_image
        

    def update(self, delta_t):
        self.mode.update(delta_t)
        if self.mode.current_mode is scattering_mode:
            self.scatter()
            #self.image = self.default_image
        elif self.mode.current_mode is chasing_mode:
            self.chase()
            #self.image = self.default_image
        MovingObject.update(self, delta_t)

    def scatter(self):
        self.goal = CustomVector()
        #self.image = self.default_image

    def chase(self):
        self.goal = self.pacman.position
        #self.image = self.default_image

    def respawn(self):
        self.goal = self.respawn_target.position
        #self.image = self.dead_image

    def setRespawnTarget(self, node):
        self.respawn_target = node

    def startRespawning(self):
        self.mode.setRespawnMode()
        if self.mode.current_mode == respawning_mode:
            self.setSpeed(150)
            self.moving_method = self.toTargetDirection
            self.respawn()
        #self.image = self.dead_image

    def startScaring(self):
        self.mode.setScaredMode()
        if self.mode.current_mode == scared_mode:
            self.setSpeed(50)
            self.moving_method = self.randomMove
            #self.image = self.scared_image

    def startNormalMode(self):
        self.setSpeed(100)
        self.moving_method = self.toTargetDirection
        self.homeNode.removePath(down_direction, self)
        #self.image = self.default_image

############################################## Blinky
class Blinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = blinky_indicator
        self.color = 'red'
        self.image = blinky_icon
        #self.default_image = blinky_icon
    
    def scatter(self):
        self.goal = CustomVector(window_width, 0)

    def chase(self):
        self.goal = self.pacman.position

############################################## Pinky
class Pinky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = pinky_indicator
        self.color = 'pink'
        self.image = pinky_icon
        #self.default_image = pinky_icon

    def scatter(self):
        self.goal = CustomVector(0, 0)

    def chase(self):
        self.goal = self.pacman.position + self.pacman.directions[self.pacman.direction] * cell_side_size * 2

############################################## Inky
class Inky(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = inky_indicator
        self.color = 'teal'
        self.image = inky_icon
        #self.default_image = inky_icon

    def scatter(self):
        self.goal = CustomVector(window_width, window_height)

    def chase(self):
        vector_1 = self.pacman.position + self.pacman.directions[self.pacman.direction] * cell_side_size * 2
        vector_2 = (vector_1 - self.blinky.position) * 2
        self.goal = self.blinky.position + vector_2

############################################## Clyde
class Clyde(Ghost):
    def __init__(self, node, pacman=None, blinky=None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = clyde_indicator
        self.color = 'orange'
        self.image = clyde_icon
        #self.default_image = clyde_icon

    def scatter(self):
        self.goal = CustomVector(0, window_height)

    def chase(self):
        distance_squared = (self.pacman.position - self.position).magnitudeSquared()
        if distance_squared <= (cell_side_size * 8)**2:
            self.scatter()
        else:
            self.goal = self.pacman.position

############################################## Ghost Group
class GhostGroup(object):
    def __init__(self, node, pacman):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        self.inky = Inky(node, pacman, self.blinky)
        self.clyde = Clyde(node, pacman)
        self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]

    def __iter__(self):
        return iter(self.ghosts)

    def update(self, delta_t):
        for ghost in self:
            ghost.update(delta_t)

    def startScaring(self):
        for ghost in self:
            ghost.startScaring()
        self.resetPoint()

    def setRespawnTarget(self, node):
        for ghost in self:
            ghost.setRespawnTarget(node)

    def upgradePoint(self):
        for ghost in self:
            ghost.point *= 2

    def resetPoint(self):
        for ghost in self:
            ghost.point = 200

    def reset(self):
        for ghost in self:
            ghost.reset()

    def render(self, screen):
        for ghost in self:
            ghost.render(screen)

    def hide(self):
        for ghost in self:
            ghost.visible = False

    def show(self):
        for ghost in self:
            ghost.visible = True



############################################## Mode
class Modes(object):
    def __init__(self):
        self.timer = 0
        self.chase()

    def update(self, delta_t):
        self.timer += delta_t
        if self.timer >= self.time:
            if self.mode is scattering_mode:
                self.chase()
            elif self.mode is chasing_mode:
                self.scatter()

    def scatter(self):
        self.mode = scattering_mode
        self.time = 7
        self.timer = 0

    def chase(self):
        self.mode = chasing_mode
        self.time = 20
        self.timer = 0

############################################## Mode Controller
class SwitchMode(object):
    def __init__(self, single_object):
        self.timer = 0
        self.time = None
        self.mainmode = Modes()
        self.current_mode = self.mainmode.mode
        self.single_object = single_object

    def update(self, delta_t):
        self.mainmode.update(delta_t)
        if self.current_mode is scared_mode:
            self.timer += delta_t
            if self.timer >= self.time:
                self.time = None
                self.single_object.startNormalMode()
                self.current_mode = self.mainmode.mode
        elif self.current_mode in [scattering_mode, chasing_mode]:
            self.current_mode = self.mainmode.mode

        if self.current_mode is respawning_mode:
            if self.single_object.node == self.single_object.respawn_target:
                self.single_object.startNormalMode()
                self.current_mode = self.mainmode.mode

    def setScaredMode(self):
        if self.current_mode in [scattering_mode, chasing_mode]:
            self.timer = 0
            self.time = 7
            self.current_mode = scared_mode
        elif self.current_mode is scared_mode:
            self.timer = 0

    def setRespawnMode(self):
        if self.current_mode is scared_mode:
            self.current_mode = respawning_mode



############################################## Gold
class Gold(MovingObject):
    def __init__(self, node, level=0):
        MovingObject.__init__(self, node)
        self.name = gold_indicator
        self.color = 'green'
        self.lifespan = 20
        self.timer = 0
        self.destroy = False
        self.point = 500 + level*100
        self.middleOfNodes(right_direction)
        self.image = gold_icon

    def update(self, delta_t):
        self.timer += delta_t
        if self.timer >= self.lifespan:
            self.destroy = True



############################################## Pause
class Pause(object):
    def __init__(self, pause=False):
        self.pause = pause
        self.timer = 0
        self.pause_duration = None
        self.function = None
        
    def update(self, delta_t):
        if self.pause_duration is not None:
            self.timer += delta_t
            if self.timer >= self.pause_duration:
                self.timer = 0
                self.pause = False
                self.pause_duration = None
                return self.function
        return None

    def setPause(self, playerPaused=False, pause_duration=None, function=None):
        self.timer = 0
        self.function = function
        self.pause_duration = pause_duration
        self.flip()

    def flip(self):
        self.pause = not self.pause



############################################## Sprite: delete
class Spritesheet(object):
    def __init__(self):
        self.sheet = pygame.image.load("spritesheet.png").convert()
        transcolor = self.sheet.get_at((0,0))
        self.sheet.set_colorkey(transcolor)
        width = int(self.sheet.get_width() / cell_side_size * cell_side_size)
        height = int(self.sheet.get_height() / cell_side_size * cell_side_size)
        self.sheet = pygame.transform.scale(self.sheet, (width, height))
        
    def getImage(self, x, y, width, height):
        x *= cell_side_size
        y *= cell_side_size
        self.sheet.set_clip(pygame.Rect(x, y, width, height))
        return self.sheet.subsurface(self.sheet.get_clip())

############################################## Maze Sprit: delete
class MazeSprites(Spritesheet):
    def __init__(self, mazefile, rotfile):
        Spritesheet.__init__(self)
        self.map = self.loadMaze(mazefile)
        self.rotdata = self.loadMaze(rotfile)

    def getImage(self, x, y):
        return Spritesheet.getImage(self, x, y, cell_side_size, cell_side_size)

    def loadMaze(self, mazefile):
        return np.loadtxt(mazefile, dtype='<U1')

    def constructBackground(self, background, y):
        for row in list(range(self.map.shape[0])):
            for col in list(range(self.map.shape[1])):
                if self.map[row][col].isdigit():
                    x = int(self.map[row][col]) + 12
                    sprite = self.getImage(x, y)
                    rotval = int(self.rotdata[row][col])
                    sprite = self.rotate(sprite, rotval)
                    background.blit(sprite, (col*cell_side_size, row*cell_side_size))
                elif self.map[row][col] == '=':
                    sprite = self.getImage(10, 8)
                    background.blit(sprite, (col*cell_side_size, row*cell_side_size))

        return background

    def rotate(self, sprite, value):
        return pygame.transform.rotate(sprite, value*90)



############################################## Maze Base
class MazeStructure(object):
    def __init__(self):
        self.portal_pairs = {}
        self.homeoffset = (0, 0)
        self.ghostNodeDeny = {up_direction:(), down_direction:(), left_direction:(), right_direction:()}

    def setPortalPairs(self, nodes):
        for pair in list(self.portal_pairs.values()):
            nodes.conncetPortalCells(*pair)

    def connectBoxNodes(self, nodes):
        key = nodes.createBox(*self.homeoffset)
        nodes.connectBoxNodes(key, self.homenodeconnectLeft, left_direction)
        nodes.connectBoxNodes(key, self.homenodeconnectRight, right_direction)

    def addOffset(self, x, y):
        return x+self.homeoffset[0], y+self.homeoffset[1]

    def denyGhostsAccess(self, ghosts, nodes):
        nodes.blockedPathList(*(self.addOffset(2, 3) + (left_direction, ghosts)))
        nodes.blockedPathList(*(self.addOffset(2, 3) + (right_direction, ghosts)))

        for direction in list(self.ghostNodeDeny.keys()):
            for values in self.ghostNodeDeny[direction]:
                nodes.blockedPathList(*(values + (direction, ghosts)))

############################################## Maze 1
class Maze1(MazeStructure):
    def __init__(self):
        MazeStructure.__init__(self)
        self.name = "maze1"
        self.portal_pairs = {0:((0, 17), (27, 17))}
        self.homeoffset = (11.5, 14)
        self.homenodeconnectLeft = (12, 14)
        self.homenodeconnectRight = (15, 14)
        self.pacman_initial_node = (15, 26)
        self.gold_initial_position = (9, 20)
        self.ghostNodeDeny = {up_direction:((12, 14), (15, 14), (12, 26), (15, 26)), left_direction:(self.addOffset(2, 3),),
                              right_direction:(self.addOffset(2, 3),)}

############################################## Maze Data
class LoadMaze(object):
    def __init__(self):
        self.object = None
        self.map_number = {0:Maze1}

    def loadMaze(self, level):
        self.object = self.map_number[level%len(self.map_number)]()



############################################## Run
class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(window_frame_size, 0, 32)
        self.background = None
        self.background_norm = None
        self.clock = pygame.time.Clock()
        self.gold = None
        self.pause = Pause(False)
        self.level = 0
        self.lives = 3
        self.score = 0

        self.gold_collected = []
        self.gold_node = None
        self.map_data = LoadMaze()

        self.font=pygame.font.Font('freesansbold.ttf', 20)

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(window_frame_size).convert()
        self.background_norm.fill('black')
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.level%5)
        self.background = self.background_norm
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
        for idx, pellet in enumerate(self.objectives.objectivesList):
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
            if ghost[1].mode.current_mode is not scared_mode and ghost[1].mode.current_mode is not respawning_mode:
                self.state[y][x] = -6
            else:
                self.state[y][x] = 6
        # dist = math.sqrt((self.pacman_prev.x - x)**2 + (self.pacman_prev.y - x)**2)
        # if abs(self.pacman_prev.x - x) >= 16 or abs(self.pacman_prev.y - y) >= 16:
        #     self.pacman_prev = self.pacman.position
        #     print("move",self.pacman.position)

        return self.state[3:34, :]
    def playGame(self):      
        self.map_data.loadMaze(self.level)
        self.mazesprites = MazeSprites(self.map_data.object.name+".txt", self.map_data.object.name+"_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup(self.map_data.object.name+".txt")
        self.map_data.object.setPortalPairs(self.nodes)
        self.map_data.object.connectBoxNodes(self.nodes)
        self.pacman = Pacman(self.nodes.nodeFromCells(*self.map_data.object.pacman_initial_node))
        self.objectives = ObjectiveGroup(self.map_data.object.name+".txt")
        self.ghosts = GhostGroup(self.nodes.startingNode(), self.pacman)

        self.ghosts.blinky.initialNode(self.nodes.nodeFromCells(*self.map_data.object.addOffset(2, 0)))
        self.ghosts.pinky.initialNode(self.nodes.nodeFromCells(*self.map_data.object.addOffset(2, 3)))
        self.ghosts.inky.initialNode(self.nodes.nodeFromCells(*self.map_data.object.addOffset(0, 3)))
        self.ghosts.clyde.initialNode(self.nodes.nodeFromCells(*self.map_data.object.addOffset(4, 3)))
        self.ghosts.setRespawnTarget(self.nodes.nodeFromCells(*self.map_data.object.addOffset(2, 3)))
        
        self.ghosts.inky.startNode.removePath(right_direction, self.ghosts.inky)
        self.ghosts.clyde.startNode.removePath(left_direction, self.ghosts.clyde)
        self.map_data.object.denyGhostsAccess(self.ghosts, self.nodes)

    def update(self):
        delta_t = self.clock.tick(120) / 1000.0
        self.objectives.update(delta_t)
        if not self.pause.pause:
            self.ghosts.update(delta_t)      
            if self.gold is not None:
                self.gold.update(delta_t)
            self.gettingObjective()
            self.collidedWithGhost()
            self.gettingGold()

        if self.pacman.alive:
            if not self.pause.pause:
                self.pacman.update(delta_t)
        else:
            self.pacman.update(delta_t)

        after_pause = self.pause.update(delta_t)
        if after_pause is not None:
            after_pause()
        self.newInput()
        self.render()
        state = self.get_frame()
        print("s1")
    def perform_action(self, action):
        state = None
        invalid_move = False
        delta_t = self.clock.tick(120) / 1000.0
        self.objectives.update(delta_t)
        if not self.pause.pause:
            self.ghosts.update(delta_t)      
            if self.gold is not None:
                self.gold.update(delta_t)
            self.gettingObjective()
            self.collidedWithGhost()
            self.gettingGold()

        if self.pacman.alive:
            if not self.pause.pause:
                self.pacman.update(delta_t,action=action)
        else:
            self.pacman.update(delta_t)

        after_pause = self.pause.update(delta_t)
        if after_pause is not None:
            after_pause()
        self.newInput()
        info = GameState()
        info.lives = self.lives
        info.invalid_move = invalid_move
        # info.total_pellets = len(
        #     self.pellets.pelletList) + len(self.eatenPellets)
        # info.collected_pellets = len(self.eatenPellets)
        info.total_pellets =100
        info.collected_pellets = 10
        state = self.get_frame()
        self.render()
        return (state, self.score, self.lives == 0 or (len(self.objectives.objectivesList) == 0), info)
    def newInput(self):
        for input in pygame.event.get():
            if input.type == QUIT:
                exit()
            elif input.type == KEYDOWN:
                if input.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.pause:
                            self.displayObjects()

    def gettingObjective(self):
        objective = self.pacman.collectObjectives(self.objectives.objectivesList)
        if objective:
            self.objectives.n_reached_objectives += 1
            self.updateScore(objective.point)
            if self.objectives.n_reached_objectives == 30:
                self.ghosts.inky.startNode.addPath(right_direction, self.ghosts.inky)
            if self.objectives.n_reached_objectives == 70:
                self.ghosts.clyde.startNode.addPath(left_direction, self.ghosts.clyde)
            self.objectives.objectivesList.remove(objective)
            if objective.name == powerups_indicator:
                self.ghosts.startScaring()

    def collidedWithGhost(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current_mode is scared_mode:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.point)                  
                    self.ghosts.upgradePoint()
                    self.pause.setPause(pause_duration=1, function=self.displayObjects)
                    ghost.startRespawning()
                elif ghost.mode.current_mode is not respawning_mode:
                    if self.pacman.alive:
                        self.lives -=  1
                        self.pacman.die()
                        self.ghosts.hide()
                        if self.lives <= 0:
                            self.pause.setPause(pause_duration=1, function=self.restartGame)
                        else:
                            self.pause.setPause(pause_duration=1, function=self.resetLevel)
    
    def gettingGold(self):
        if self.objectives.n_reached_objectives == 10 or self.objectives.n_reached_objectives == 100 or self.objectives.n_reached_objectives == 150:
            if self.gold is None:
                self.gold = Gold(self.nodes.nodeFromCells(9, 20), self.level)
        if self.gold is not None:
            if self.pacman.collideCheck(self.gold):
                self.updateScore(self.gold.point)
                gold_collected = False
                for gold in self.gold_collected:
                    if gold.get_offset() == self.gold.image.get_offset():
                        gold_collected = True
                        break
                if not gold_collected:
                    self.gold_collected.append(self.gold.image)
                self.gold = None
            elif self.gold.destroy:
                self.gold = None

    def displayObjects(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideObjects(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def restartGame(self):
        self.lives = 3
        self.level = 0
        self.gold = None
        self.playGame()
        self.score = 0
        self.gold_collected = []

    def resetLevel(self):
        self.pacman.reset()
        self.ghosts.reset()
        self.gold = None

    def updateScore(self, point):
        self.score += point

    
    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.objectives.render(self.screen)
        if self.gold is not None:
            self.gold.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)

        for i in range(self.lives):
            x = heart_icon.get_width() * i + 20
            y = window_height - heart_icon.get_height()
            self.screen.blit(heart_icon, (x, y))
            
        score_text = self.font.render(f'Score:{self.score}',True,'white')
        self.screen.blit(score_text, (window_width-125, window_height-25))

        pygame.display.update()



if __name__ == "__main__":
    game = GameController()
    game.playGame()
    while True:
        game.update()

