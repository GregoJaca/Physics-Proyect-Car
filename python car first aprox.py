import numpy as np
import math
import pygame
from pygame.locals import *
import sys
from PIL import Image

#________________GLOBAL_VAR_____________
population_n = 100
selection_n = 60 # min 50
mutation_rate = 0

population = []

#________________PYGAME_VARS____________
HEIGHT = 648
WIDTH = 1152
FPS = 120

grey = pygame.Color(100,100,100)
black = pygame.Color(0,0,0)
white = pygame.Color(255,255,255)
green = pygame.Color(0,100,0)

race_track = pygame.image.load("map1.jpg")

# -----------------ROAD------------------
road_im = np.array(Image.open("map1.jpg"))
road = np.empty((HEIGHT, WIDTH))

for i in range(HEIGHT):
    for j in range(WIDTH):
        if road_im[i,j,0] == 150 and road_im[i,j,1] == 150 and road_im[i,j,2] == 150:
          road[i][j] = 1
        else:
            road[i][j] = -1
    

# -----------------------CAR-------------------

# parameters

velpen = 0.1 #if outside the road, it reduces the speed by 90%
dt = 0.1
car_size = 13
#how far away it can see


sightdist = 20
sightnum = 5
#number of eyes. minumum 3
numeyes = 5
nn_size = [numeyes*sightnum,2]

#max speed
max_speed = 10
max_truning_speed = 0.5
acc_multiplier = 5


#parameters for Fitness
maxcount = 100 #how many steps before calculating dist traveled
weightdist = 1 #how important is dist traveled for fitness
weightroad = -25 #how important is being on road for fitness

# initial conditions
posi =  np.array([125,465] )
veli =  np.array([0,0] )
anglei = 0


class Car(pygame.sprite.Sprite):

    def __init__(self):
        #for each car
        super().__init__() 
        self.surf = pygame.Surface((car_size, car_size+10), pygame.SRCALPHA)
        self.orig_surf = self.surf
        self.surf.fill((128,255,40))
        self.rect = self.surf.get_rect(center = (posi[0], posi[1]))

        self.pos = posi
        self.vel = veli
        self.angle = 0
        self.w = 0 # w is the derivative of angle with respect to time
        
        #for each car's wheels
        #each wheel has a scalar number which indicates its speed and acc. the first number is for left wheel and second for right wheel
        self.wheelvel = np.array([0,0])
        self.wheelacc = np.array([0,0])
        #for fitness
        self.count = 0
        self.fitness = 0
        self.savedpos = posi

        #cars NN
        self.NN = NN(nn_size)

    def actualize(self):  

        #NN controls wheelvel
        vision_input = self.see()*acc_multiplier
        self.wheelacc = self.NN.predict(vision_input)

        #euler method of integration.

        #wheelacc should be calculated by the neural network
        self.wheelvel = self.wheelvel + self.wheelacc * dt

        if self.wheelvel[0] < 0:
            self.wheelvel[0] = 0
        if self.wheelvel[1] < 0:
            self.wheelvel[1] = 0

        #euler for angle
        self.w = (self.wheelvel[0] - self.wheelvel[1]) / car_size
        if self.w > max_truning_speed:
            self.w = max_truning_speed
        if self.w < -max_truning_speed:
            self.w = -max_truning_speed
        
        self.angle = self.angle + self.w * dt

        #angle should stay between 0-2pi
        if self.angle >= np.pi*2:
            self.angle -= np.pi*2
        
        if self.angle < 0:
            self.angle += np.pi*2

        #vel of the car is the average of both wheels velocity (and multiplied by the unit vector with the angle of the car)
        speed = (self.wheelvel[0] + self.wheelvel[1]) / 2 

        if speed > max_speed:
            speed = max_speed
        
        self.vel =  speed * np.array( [math.sin(self.angle), math.cos(self.angle)] )

        #Here if that wheel is outside the road, we make the vel smaller as punishment
        if self.onRoad() != 1:
            self.vel *= velpen

        # euler for position
        self.pos = self.pos + self.vel * dt

        # drawing new position
        self.surf = pygame.transform.rotate(self.orig_surf, np.rad2deg(self.angle))
        self.rect = self.surf.get_rect(center = self.pos)
        self.rect.center = self.pos

    def calcFitness(self):
        #calc fitness should be called together with actualize
        #every maxcount steps it calculates how much distance it traveled and saves a new position as starting point
        if self.count == maxcount:
            self.fitness += weightdist * np.linalg.norm( self.pos - self.savedpos )
            self.savedpos = self.pos
            self.count = 0
        self.count += 1

        if self.onRoad() != 1:
            self.fitness += weightroad
    #as input it gets which eye is looking, and returns an array with the values of the road in that direction
    #when you call the function you should call it in a loop for i in range(numeyes)
    def see(self): 

        view = np.empty(sightnum*numeyes)

        for i in range(numeyes):
            direction = self.angle - math.pi / 2 + i * math.pi / (numeyes-1)

            if direction >= np.pi*2:
                direction -= np.pi*2
            if direction < 0:
                direction += np.pi*2
    
            for j in range(sightnum):
                view[i*sightnum+j] = road[ round( self.pos[0] + math.cos(direction) * sightdist * j) , round( self.pos[1] +  math.sin(direction) * sightdist * j) ]

        return view


# this should return the same value as the value of the road at that point (a 0 (or false) if outside the road and viceversa)
    def onRoad(self):
        return road[ round(self.pos[0]) , round(self.pos[1]) ] #pos can be a float, but to check we round
    

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
    parents = []

    for i in range(population_n):
        fitness_arr[i] = population[i].fitness
        if fitness_arr[i] < 0:
            fitness_arr[i] = 0
        print(fitness_arr[i])

    random_choices = np.random.choice(population_n,selection_n,True,fitness_arr/sum(fitness_arr))
    
    for i in range(population_n*2):
        parents.append(population[random_choices[i % selection_n]])

    return parents


def give_birth(parents): #I had to name it this

    new_gen = []
    layers_n = len(nn_size)

    
    #baby creation
    for i in range(population_n):
        baby = Car()
        for j in range(layers_n-1):
            weight_genes = np.random.randint(2,size = (nn_size[j+1],nn_size[j]))
            bias_genes = np.random.randint(2,size = nn_size[j+1])
            baby.NN.layers[j].weights = (np.multiply(parents[i*2].NN.layers[j].weights, weight_genes) + np.multiply(parents[i*2+1].NN.layers[j].weights, np.ones((nn_size[j+1], nn_size[j])) - weight_genes))
            baby.NN.layers[j].biases = (np.multiply(parents[i*2].NN.layers[j].biases,bias_genes) + np.multiply(parents[i*2+1].NN.layers[j].biases, np.ones(nn_size[j+1]) - bias_genes))

            # Mutating. We either use this or the mutate() function
            # this has the limitation that it can only modify one "gene" of each individual
            # using the mutate() function, the same individual could be picked many times and get many genes modified
            if np.random.rand() < mutation_rate:
                a = np.random.binomial(1 , nn_size[j+1] / nn_size[j])
                if a == 1:
                    #only one biases is changed
                    baby.layers[j].biases[ np.random.choice(nn_size[j+1]) ] = np.random.rand() * 2 - 1
                    #all biases are changed
                    #baby.layers[j].biases = np.random.rand(nn_size[j+1])
                else:
                    #only one weight is changed
                    baby.layers[j].weights[ np.random.choice(nn_size[j]) ] = np.random.rand() * 2 - 1
                    #all weights are changed
                    #baby.layers[j].weights = np.random.rand(nn_size[j])
        
        new_gen.append(baby) 
        
    return new_gen


def mutate(population):

    mut_number_avg = 50 #this should be a Global variable. I just wrote it here to be clear. Then I'll move it.
    mut_number = np.random.binomial(mut_number_avg * 2 , 0.5)

    for i in range(mut_number):
        mut_ind = np.random.choice(population_n)
        mut_layer = np.random.choice( len(nn_size) )

        if np.random.binomial(1 , nn_size[1] / nn_size[0]) == 1:
            #mutate biases
            population[mut_ind].layers[mut_layer].biases[ np.random.choice(nn_size[j+1]) ] = np.random.rand() * 2 - 1
        else: 
            #mutate weights
            population[mut_ind].layers[mut_layer].weights[ np.random.choice(nn_size[j+1]) ] = np.random.rand() * 2 - 1
            
    return population

#creating population #1 for testing
for i in range(population_n):
    population.append(Car())


#pygame initialization
pygame.init()
FramePerSec = pygame.time.Clock()
 
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game")

all_sprites = pygame.sprite.Group()

#creating population #1 for testing
for i in range(population_n):
    population[i] = Car()
    all_sprites.add(population[i])
    


#main loop
while True:

    #exit
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    displaysurface.fill(green)
    displaysurface.blit(race_track,(0,0))

    for i in range(population_n):
        population[i].actualize()
        population[i].calcFitness()
    for entity in all_sprites:
        displaysurface.blit(entity.surf, entity.rect)

    pygame.display.update()
    FramePerSec.tick(FPS)


    #when space pressed create new generation
    pressed_keys = pygame.key.get_pressed()
    if pressed_keys[K_SPACE]:
        population = give_birth(selection(population))
        # population = mutate(population)
        displaysurface.fill(green)
        displaysurface.blit(race_track,(0,0))
        displaysurface.blit(entity.surf, entity.rect)
        for entity in all_sprites:
            entity.kill()
        for i in range(population_n):
            all_sprites.add(population[i])

        FramePerSec.tick(1)


