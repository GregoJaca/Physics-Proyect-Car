import numpy as np
import math
import pygame



# -----------------ROAD------------------

width = 21
height = 21
centerx = round(width / 2)
centery = round(height / 2)

#make rad similar to centerx and centery
rad = 9
epsilon = 2.5

#first create map with all 0 (outside the road)
road = np.empty([width, height])

# draw the circle. 1 if its part of the road. 0 if it is not part of the road
for y in range(height):
    for x in range(width):
        # see if we're close to (x-a)**2 + (y-b)**2 == r**2
        if abs((x - centerx)**2 + (y - centery)**2 - rad**2) < epsilon**2:
            road[y][x] = 1
        else:
            road[y][x] = 0

# print the map
print(road)

# -----------------------CAR-------------------

# parameters
velpen = 0.8 #if outside the road, it reduces the speed by 20%
dt = 0.01
# D is the size of the car
D = 1
#parameters for Fitness
maxcount = 5 #how many steps before calculating dist traveled
weightdist = 1 #how important is dist traveled for fitness
weightroad = -0.1 #how important is being on road for fitness

# initial conditions
posi =  [(width - D) / 2, rad] 
veli =  [0,0] 
anglei = 0


class Car:

    def __init__(self):
        #for each car
        self.pos = posi
        self.vel = veli
        self.angle = 0
        self.w = 0 # w is the derivative of angle with respect to time
        
        #for each car's wheels
        #each wheel has a scalar number which indicates its speed and acc. the first number is for left wheel and second for right wheel
        self.wheelvel = [0 , 0]
        self.wheelacc = [0,0]

        #for fitness
        self.count = 0
        self.fitness = 0
        self.savedpos = posi

        #cars NN
        self.NN = NN([5,6,2])

    def actualize(self):  

        #euler method of integration.

        #wheelacc should be calculated by the neural network
        self.wheelvel += self.wheelacc * dt

        #euler for angle
        self.w = (self.wheelvel[0] + self.wheelvel[1]) / D
        self.angle += self.w * dt

        #vel of the car is the average of both wheels velocity (and multiplied by the unit vector with the angle of the car)
        self.vel = ( (self.wheelvel[0] + self.wheelvel[1]) / 2 ) * [math.cos(self.angle), math.sin(self.angle) ]

        #Here if that wheel is outside the road, we make the vel smaller as punishment
        if not self.onRoad():
            self.vel *= velpen

        # euler for position
        self.pos += self.vel * dt

    def calcFitness(self):
        #calc fitness should be called together with actualize
        #every maxcount steps it calculates how much distance it traveled and saves a new position as starting point
        if self.count == maxcount:
            self.fitness += weightdist * np.linalg.norm( self.pos - self.savedpos )
            self.savedpos = self.pos
        count += 1

        if not self.onRoad():
            self.fitness += weightroad

# this should return the same value as the value of the road at that point (a 0 (or false) if outside the road and viceversa)
    def onRoad(self):
        return road[self.pos]
    

#Neural Network Class
class NN():

    def __init__(self,sizes):
        self.sizes = sizes
        self.layers = []
        for i in range(len(self.sizes)-1):
                self.layers.append(FCL(sizes[i],sizes[i+1]))
        
    #predicts 
    def predict(self,inp):
        for layer in self.layers:
            layer.inp = inp
            layer.out = np.tanh(layer.weights.dot(layer.inp)+layer.biases)
            inp = layer.out

        return layer.out   

#Class of a NN layer
class FCL():

    #initialized with random biases and weights
    def __init__(self,in_size,out_size):
        self.inp = np.zeros(in_size)
        self.out = np.zeros(out_size)
        self.weights = np.random.rand(out_size,in_size)*2-1
        self.biases = np.random.rand(out_size)*2-1

#creating and testing the neural network, later it can be deleted it is here to show example
neural_net = NN([5,6,2])
print(neural_net.predict(np.array([0.2,0,0.5,0,0])))


# what prof told me: have acc and vel for wheels. 
# have (angle, pos, vel) for car. then, vel car is the average. w = v2 - v1 / (D/2) (grego: it is / D). angle += w . dt
# so, having vel and angle for the car, we calculate vector velocity for car. we integrate using euler method

# for the fitness. we should measure distance traveled every n steps (n = 5) 
# using distances greater than D to prevent giving high fitness to cars rotating.
# also take into considerantion if the car is on or outside the road.

