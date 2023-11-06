import pygame
from pygame.locals import *
import sys
import numpy as np
 
pygame.init()

vec = pygame.math.Vector2  # 2 for two dimensional
 

ACC = 0.5
AACC = 5
FRIC = -0.12
 
HEIGHT = 450
WIDTH = 400
FPS = 60
 
FramePerSec = pygame.time.Clock()
 
displaysurface = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game")

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__() 
        self.surf = pygame.Surface((30, 30),pygame.SRCALPHA)
        self.orig_surf = self.surf
        self.surf.fill((128,255,40))
        self.rect = self.surf.get_rect(center = (10, 420))

        self.pos = vec((10, 385))
        self.vel = 0
        self.acc = 0

        self.ang = 0

    def move(self):
        self.acc = 0
 
        pressed_keys = pygame.key.get_pressed()
            
        if pressed_keys[K_UP]:
            self.acc = -ACC
        if pressed_keys[K_DOWN]:
            self.acc = ACC 
        
        if pressed_keys[K_LEFT]:
    
            self.ang += AACC
            if self.ang >= 360:
                self.ang-=360
            
        
        if pressed_keys[K_RIGHT]:
            
            self.ang -= AACC
            if self.ang < 0:
                self.ang+=360
            
        

        self.acc += self.vel * FRIC
        self.vel += self.acc
        self.pos.x += (self.vel + 0.5 * self.acc)*np.sin(np.deg2rad(self.ang))
        self.pos.y += (self.vel + 0.5 * self.acc)*np.cos(np.deg2rad(self.ang))

        if self.pos.x > WIDTH:
            self.pos.x = 0
        if self.pos.x < 0:
            self.pos.x = WIDTH

        self.surf = pygame.transform.rotate(self.orig_surf, self.ang)
        self.rect = self.surf.get_rect(center = self.pos)
        self.rect.center = self.pos
 
P1 = Player()
P2 = Player()

all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
all_sprites.add(P2)
 
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
     
    displaysurface.fill((0,0,0))

    P1.move()
 
    for entity in all_sprites:
        displaysurface.blit(entity.surf, entity.rect)
 
    pygame.display.update()
    FramePerSec.tick(FPS)