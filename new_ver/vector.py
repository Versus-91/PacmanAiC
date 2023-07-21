#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
class Vectors(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.thresh = 0.000001 #
    def __add__(self, other):
        return Vectors(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vectors(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vectors(-self.x, -self.y)

    def __mul__(self, scalar):
        return Vectors(self.x * scalar, self.y * scalar)

    def __div__(self, scalar):
        if scalar != 0:
            return Vectors(self.x / float(scalar), self.y / float(scalar))
        return None

    def __truediv__(self, scalar):
        return self.__div__(scalar)
    def __eq__(self, other):
        if abs(self.x - other.x) < self.thresh:
            if abs(self.y - other.y) < self.thresh:
                return True
        return False
    def __str__(self): #print results
        return "<"+str(self.x)+", "+str(self.y)+">"
    
    def magnitudeSquared(self): #rename
        return self.x**2 + self.y**2

    def lenght(self): #magnitude
        return math.sqrt(self.magnitudeSquared())
    def copy(self):
        return Vectors(self.x, self.y)

    def asTuple(self): #rename
        return self.x, self.y

    def asInt(self):#rename
        return int(self.x), int(self.y)
    

