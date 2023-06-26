SCATTER = 0
CHASE = 1
FREIGHT = 2
SPAWN = 3
#from ghost1 import Ghost
class MainMode(object):
    def __init__(self):
        self.timer = 0
        self.scatter()

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.time:
            if self.mode is SCATTER:
                self.chase()
            elif self.mode is CHASE:
                self.scatter()

    def scatter(self):
        self.mode = SCATTER
        self.time = 7
        self.timer = 0

    def chase(self):
        self.mode = CHASE
        self.time = 20
        self.timer = 0
        

class ModeController(object):
    def __init__(self, Ghost):
        self.timer = 0
        self.time = None
        self.mainmode = MainMode()
        self.current = self.mainmode.mode
        self.ghost = Ghost 
    
    def update(self, dt):
        self.mainmode.update(dt)
        if self.current is FREIGHT:
            self.timer += dt
            if self.timer >= self.time:
                self.time = None
                self.ghost.normalMode()
                self.current = self.mainmode.mode
        #else:
         #   self.current = self.mainmode.mode
        elif self.current in [SCATTER, CHASE]:
        
            self.current = self.mainmode.mode

        if self.current is SPAWN:
            if self.ghost.node == self.ghost.spawnNode:
                self.ghost.normalMode()
                self.current = self.mainmode.mode

    def setSpawnMode(self):
        if self.current is FREIGHT:
            self.current = SPAWN
    def setFreightMode(self):
        if self.current in [SCATTER, CHASE]:
            self.timer = 0
            self.time = 7   #change how long is frozen
            self.current = FREIGHT
        elif self.current is FREIGHT:
            self.timer = 0
        