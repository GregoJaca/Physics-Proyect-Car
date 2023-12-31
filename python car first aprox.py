import numpy as np
import math
from scipy.spatial import distance_matrix
import pygame
from pygame.locals import *
import sys
from PIL import Image

#________________GLOBAL_VAR_____________
population_n = 100
selection_n = 60 # min 50
mutation_rate = 0.9 # between 0 and 1

gen_diversity_low_limit = 1 #no idea how much it should be

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

velpen = 0.8 #if outside the road, it reduces the speed by 20% every time
dt = 0.2
car_size = 13
sightdist = 5 #how far away it can see. // I'm not sure but I think that the furthest it can see is sightdist * sightnum
sightnum = 5
#number of eyes. minumum 3
numeyes = 5
nn_size = [numeyes*sightnum,2]
layers_n = len(nn_size)

#max speed
max_speed = 10
max_truning_speed = 0.5
acc_multiplier = 5


#parameters for Fitness
maxcount = 150 #how many steps before calculating dist traveled
weightdist = 20 #how important is dist traveled for fitness
weightroad = -20 #how important is being on road for fitness
weighcrash = -10

# initial conditions
posi =  np.array([125,465] )
veli =  np.array([0,0] )
anglei = 0


class Car(pygame.sprite.Sprite):

    def __init__(self):
        #for each car
        super().__init__() 
        self.surf = pygame.Surface((car_size, car_size), pygame.SRCALPHA)
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

        # if self.is_crashing(population):
        # self.fitness += weighcrash

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
    
    # Because they all start at the same position and there are SO many cars, it's a bit weird.
    # We can use it if we have only few cars learning how to drive without crashing into each other
    def is_crashing(self, population): 
        for i in range(population_n):
            if ( (np.all(population[i].pos != self.pos) ) and (np.linalg.norm(population[i].pos - self.pos) <  car_size)):
                return True
            
        return False

    

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

#GENETIC_ALGORITHM_FUNCTIONS_________________________________________________________________________

avg_fitness = []
#list of cars as the parameter
def selection(population):

    fitness_arr = np.zeros(population_n)
    parents = []

    for i in range(population_n):
        fitness_arr[i] = population[i].fitness
        if fitness_arr[i] < 0:
            fitness_arr[i] = 0
        # print(fitness_arr[i])
    avg_fitness.append( sum(fitness_arr) / population_n )

    if sum(fitness_arr) != 0:
        random_choices = np.random.choice(population_n,selection_n,True,fitness_arr/sum(fitness_arr))
    else: 
        random_choices = np.random.choice(population_n,selection_n,True)
    for i in range(population_n*2):
        parents.append(population[random_choices[i % selection_n]])

    return parents


def give_birth(parents): #I had to name it this

    new_gen = []
    

    
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
            
            # if we want to generalize to more NN layers, should this if condition change ? 
            if np.random.rand() < mutation_rate:
                if np.random.rand() < (nn_size[j+1] / nn_size[j]):
                    baby.NN.layers[j].biases[np.random.choice(nn_size[j+1])] = np.random.rand() * 2 - 1
                else:
                    baby.NN.layers[j].weights[np.random.choice(nn_size[j+1])][np.random.choice(nn_size[j])] = np.random.rand() * 2 - 1

        new_gen.append(baby) 
    return new_gen


# is made for one layer
# isn't finished
def mutate(population):

    mut_num_weights_avg = 40 
    mut_num_biases_avg = 10 #these 2 should be a Global variable. Wrote it here to be clear. Then I'll move it.

    # Prof wanted to use a binomial distribution for the number of mutations each gen. I think it's unnecessary
    mut_num_weights = np.random.binomial(mut_num_weights_avg * 2 , 0.5)
    mut_num_biases  = np.random.binomial(mut_num_biases_avg  * 2 , 0.5)

    # if we want to generalize to more NN layers, the nn_size[x] condition should change
    mut_individuals_weights = np.random.rand( population_n * nn_size[0] , mut_num_weights , replace = False )
    mut_individuals_biases  = np.random.rand( population_n * nn_size[1] , mut_num_biases  , replace = False )

    # Prof told me to do it this way. He told me that I could index into population[] and weights[] and biases[] using the randomly generated arrays
    # That way it wouldn't need a loop and be faster. I couldn't do it, so I used loops.
    for i in range(mut_num_weights):
        population[mut_individuals_weights[i] // nn_size[0]].NN.layers[np.random.choice( len(nn_size) -1 )].weights[ mut_individuals_weights[i] % nn_size[0] ] = np.random.rand() * 2 - 1
    
    for i in range(mut_num_biases):
        population[mut_individuals_biases[i]  // nn_size[1]].NN.layers[np.random.choice( len(nn_size) -1 )].biases [ mut_individuals_biases[i]  % nn_size[0] ] = np.random.rand() * 2 - 1

    return population


max_gen_dist_list = []
def max_genetic_distance (population): 
    genes = []
    for i in range(population_n):

        # This is for a general NN of any size

        # for j in range(layers_n-1):
        #     ind_genome = population[i].NN.layers[j].biases
        #     for k in range(layers_n):
        #         ind_genome = np.concatenate((ind_genome , population[i].NN.layers[j].weights[k]) , axis = 0)
        
        # This is simplified for 1 layer.
        ind_genome = np.concatenate((population[i].NN.layers[0].biases , population[i].NN.layers[0].weights[0], population[i].NN.layers[0].weights[1]) , axis = 0)
        
        genes.append(ind_genome)

    max_gen_dist = np.max( distance_matrix( genes, genes ) )
    max_gen_dist_list.append(max_gen_dist)

    return max_gen_dist


def max_gen_dist_derivative (n):
    # checks that list is big enough. It's just for safety, but we should take it out in the end.
    if len(max_gen_dist_list) < n and n <= 2:
        return ( gen_diversity_low_limit + 1 , 1 ) #returns values that won't stop the simulation

    first_deriv = (max_gen_dist_list[-1] - max_gen_dist_list[-n]) / n
    second_deriv = ( (max_gen_dist_list[-n + 1] - max_gen_dist_list[-n]) - (max_gen_dist_list[-1] - max_gen_dist_list[-2]) ) / (n-1)
    return ( first_deriv , second_deriv )



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


# Check this parameters to know if we should quit the simulation. Initial values won't stop the simulation.
gen_diversity_1deriv = gen_diversity_low_limit + 1
gen_diversity_2deriv = 1
generation_n = 1

#---------------main loop-------------------------
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
        max_genetic_distance(population)
        displaysurface.fill(green)
        displaysurface.blit(race_track,(0,0))
        displaysurface.blit(entity.surf, entity.rect)
        for entity in all_sprites:
            entity.kill()
        for i in range(population_n):
            all_sprites.add(population[i])
        FramePerSec.tick(1)

        if generation_n > 20:
            gen_diversity_1deriv , gen_diversity_2deriv = max_gen_dist_derivative(5) # the argument must be bigger than 2
        
        generation_n += 1
        
        

        # basically checks if gene diversity is low and decreasing
        # I'm not completely sure about the decreasing thing, because eventually it will reach a point where
        # due to mutations it will be constant. so maybe the last condition is unnecessary
        if generation_n > 20 and gen_diversity_1deriv < gen_diversity_low_limit and gen_diversity_2deriv < 0:
            pygame.quit()
            print("Finished because of low genetic diversity after ", generation_n, "generations")
            print("Gen div 1st deriv: ", gen_diversity_1deriv)
            sys.exit()

    




