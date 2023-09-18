import numpy as np
import math
# import pygame

#________________GLOBAL_VAR_____________
population_n = 10
selection_n = 5
nn_size = [5,6,2]

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
#how far away it can see
sightdist = 3 * D
#number of eyes. minumum 3
numeyes = 5
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
        self.NN = NN([numeyes,6,2])

    def actualize(self):  

        #euler method of integration.

        #wheelacc should be calculated by the neural network
        self.wheelvel += self.wheelacc * dt

        #euler for angle
        self.w = (self.wheelvel[0] + self.wheelvel[1]) / D
        self.angle += self.w * dt

        #vel of the car is the average of both wheels velocity (and multiplied by the unit vector with the angle of the car)
        self.vel = ( np.linalg.norm(self.wheelvel[0] + self.wheelvel[1]) / 2 ) * [math.cos(self.angle), math.sin(self.angle) ]

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
    
    #as input it gets which eye is looking, and returns an array with the values of the road in that direction
    #when you call the function you should call it in a loop for i in range(numeyes)
    def see(self, eye): 
        direction = self.angle - math.pi / 2 + eye * math.pi / numeyes
        view = []
        for i in sightdist:
            view.append(road[ np.round_( self.pos + i *  [math.cos(direction), math.sin(direction) ] ) ])

        return view


# this should return the same value as the value of the road at that point (a 0 (or false) if outside the road and viceversa)
    def onRoad(self):
        return road[ np.round_(self.pos) ] #pos can be a float, but to check
    

#Neural Network Class
class NN:

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
class FCL:

    #initialized with random biases and weights
    def __init__(self,in_size,out_size):
        self.inp = np.zeros(in_size)
        self.out = np.zeros(out_size)
        self.weights = np.random.rand(out_size,in_size)*2-1
        self.biases = np.random.rand(out_size)*2-1

#GENETIC_ALGORITHM_FUNCTIONS_________________________________________________________________________-

#list of cars as the parameter
def selection(population):

    fitness_arr = np.zeros(population_n)
    parents = np.zeros(population_n)

    for i in range(population_n):
        fitness_arr[i] = population[i].fitness

    random_choices = np.random.choice(population_n,selection_n,True,fitness_arr/sum(fitness_arr))
    
    for i in range(population_n):
        parents[i] = population[random_choices[i % selection_n]]

    return parents


def give_birth(parents, mutation_rate): #I had to name it this

    new_gen = np.zeros(population_n)
    layers_n = len(nn_size)


    for i in range(population_n):
        baby = Car()
        for j in range(layers_n-1):
            weight_genes = np.random.randint(2,size = (nn_size[j+1],nn_size[j]))
            weight_mutation = [] #must finish
            bias_genes = np.random.randint(2,size = nn_size[j+1])
            bias_mutation = [] #must finish
            baby.NN.layers[j].weights = np.multiply(parents[i*2].NN.layers[j].weights, weight_genes) + np.multiply(parents[i*2+1].NN.layers[j].weights, np.ones(nn_size[j+1], nn_size[j]) - weight_genes)
            baby.NN.layers[j].biases = np.multiply(parents[i*2].NN.layers[j].biases,bias_genes) + np.multiply(parents[i*2+1].NN.layers[j].biases, np.ones(nn_size[j+1]) - bias_genes)
        new_gen[i] = baby


    return new_gen

#creating and testing the neural network, later it can be deleted it is here to show example
neural_net = NN(nn_size)
print(neural_net.predict(np.array([0.2,0,0.5,0,0])))
print(selection([1,5,7,3,4,1,5,7,3,4]))


# what prof told me: have acc and vel for wheels. 
# have (angle, pos, vel) for car. then, vel car is the average. w = v2 - v1 / (D/2) (grego: it is / D). angle += w . dt
# so, having vel and angle for the car, we calculate vector velocity for car. we integrate using euler method

# for the fitness. we should measure distance traveled every n steps (n = 5) 
# using distances greater than D to prevent giving high fitness to cars rotating.
# also take into considerantion if the car is on or outside the road.
