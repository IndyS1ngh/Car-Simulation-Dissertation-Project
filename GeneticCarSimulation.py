
#Genetic Car Simulation by Inderpreet Sandhu - 852298
#References: https://rednuht.org/genetic_cars_2/ , http://boxcar2d.com/ , Pavitra Kumar's Genetic Cars Box2D and iforce2d.net.

import math
import time
import copy
import pygame
from Box2D import *
from pygame.locals import *
pygame.init()
import sys
from Box2D.b2 import *
import random
import matplotlib.pyplot as plt
import numpy as np

wheelRadMax, wheelRadMin, wheelDenMax, wheelDenMin = 0.5, 0.2, 30, 10
hullCoordMax, hullCoordMin, hullDenMax, hullDenMin = 1.1, 0.1, 100, 50
roadWidth, roadHeight = 1.5, 0.15
beginningPos = b2Vec2(1,2)
carLifePoints = 100
carAccel = 25
isRunning = True
grav = b2Vec2(0.0, -9.807)
maxDist = 0
genMaxDist = 0
carsEliminated = 0
distTracker = 0
localGenCounter = 0
genMaxDistCopy = 0
#bestSolImage = pygame.image.load("BestSolution.jpg")
seed = random.randint(1,39478534)

#Seed for testing mutation rates 
#seed = 1459152
 
maxDist1 = 0
storedDist1 = 0
window = pygame.display.set_mode((1080,800), 0, 32)
gcCopy = 0
mutationRate = 4

class Stats:
    def setHullDen(self, density):
        self.hullDen = density
    def setVertices(self, vertices):
        self.vertices = vertices
    def setWheelNum(self, number):
        self.wheelNum = number
    def setWheelRad(self, radius):
        self.wheelRad = radius
    def setWheelDen(self, density):
        self.wheelDen = density
    def setWheelVert(self, vertices):
        self.wheelVert = vertices
        
    def getHullDen(self):
        return self.hullDen
    def getVertices(self):
        return self.vertices;
    def getWheelNum(self):
        return self.wheelNum
    def getWheelRad(self):
        return self.wheelRad
    def getWheelDen(self):
        return self.wheelDen
    def getWheelVert(self):
        return self.wheelVert
    
    def __init__(self):
        self.wheelNum = 2
        self.wheelRad = [0,0]
        self.wheelDen = [0,0]
        self.wheelVert = [0,0]
        self.hullDen = 1
        self.vertices = [0,0,0,0,0,0,0,0]
        
class carVehicle:
    #genome:
        #wheel num (1 gene)
        #wheel radius (2 genes, 1 per wheel)
        #wheel vertices (2 genes, 1 per wheel)
        #wheel density (2 genes, 1 per wheel)
        #hull vertices   (8 genes, 1 per vertex)
        #hull density (1 gene)
    def __init__(self, world, random=True, cDef = None):
        global wheelRadMax,wheelRadMin,wheelDenMax,wheelDenMin,hullCoordMax,hullCoordMin,hullDenMax,hullDenMin,roadWidth,roadHeight,beginningPos,carLifePoints,carAccel,isRunning,grav
        self.world = world
        if random:
            self.cDef = self.generateCar()
        else:
            self.cDef = cDef
        self.movementSpeed = 0
        self.isMoving = True
        self.hull = self.createHull(self.cDef.vertices, self.cDef.hullDen)
        self.wheels = []
        
        for i in range(self.cDef.wheelNum):
            self.wheels.append(self.generateWheel(self.cDef.wheelRad[i], self.cDef.wheelDen[i]))
        carWeight = self.hull.mass
        
        for i in range(self.cDef.wheelNum):
            carWeight += self.wheels[i].mass
        carWeight = 2+1
        
        self.speed = []
        for i in range(self.cDef.wheelNum):
            self.speed.append(carWeight * -grav.y / self.cDef.wheelRad[i])
        self.joint = b2RevoluteJointDef()
        
        for i in range(self.cDef.wheelNum):
            rVert = self.hull.vertices[self.cDef.wheelVert[i]]
            self.joint.maxMotorTorque = self.speed[i]
            self.joint.motorSpeed = -carAccel
            self.joint.enableMotor = True
            self.joint.localAnchorA.Set(rVert.x, rVert.y)
            self.joint.localAnchorB.Set(0, 0)
            self.joint.collideConnected = False
            self.joint.bodyA = self.hull
            self.joint.bodyB = self.wheels[i]
            j = self.world.CreateJoint(self.joint)
    
    def generateCar(self):
        global wheelRadMax,wheelRadMin,wheelDenMax,wheelDenMin,hullCoordMax,hullCoordMin,hullDenMax,hullDenMin,roadWidth,roadHeight,beginningPos,carLifePoints,carAccel,isRunning,grav
        genCar = Stats()
        wRad = []
        wDen = []
        vertices = []
        wVert = []
        for i in range(genCar.getWheelNum()):
            wRad.append(random.random()*wheelRadMax+wheelRadMin)
            wDen.append(random.random()*wheelDenMax+wheelDenMin)
        
        # vertices generally appended in a shape resembling this for the hull:
        # [4].         .[3]         .[2]


        # [5].         .(0,0)       .[1]


        # [6].         .[7]         .[8]
        
        vertices.append(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
        vertices.append(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
        vertices.append(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
        vertices.append(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
        vertices.append(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
        vertices.append(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
        vertices.append(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
        vertices.append(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
        
        iLeft = [i for i in range(8)]
        for i in range(genCar.getWheelNum()):
            iNext = int(random.random() * (len(iLeft)-1))
            wVert.append(iLeft[iNext])
            iLeft = iLeft[:iNext] + iLeft[iNext+1:]
        genCar.setVertices(vertices)
        genCar.setWheelRad(wRad)
        genCar.setWheelDen(wDen)
        genCar.setWheelVert(wVert)
        genCar.setHullDen(random.random()*hullDenMax + hullDenMin)
        return genCar
    
    def generateWheel(self, radius, density):
        wheelBody = bodyDef()
        wheelBody.type = b2_dynamicBody
        wheelBody.position.Set(beginningPos.x, beginningPos.y)
        wheel = self.world.CreateBody(wheelBody)
        fDef = b2FixtureDef()
        fDef.shape = b2CircleShape(radius = radius)
        fDef.density = density
        fDef.friction = 1
        fDef.restitution = 0.2
        fDef.filter.groupIndex = -1
        wheel.CreateFixture(fDef)
        return wheel
    
    def hullGen(self,hull,v1,v2,density):
        vertices = []
        vertices.append(v1)
        vertices.append(v2)
        vertices.append(b2Vec2(0,0))
        fDef = b2FixtureDef()
        fDef.shape = b2PolygonShape()
        fDef.density = density
        fDef.friction = 10
        fDef.restitution = 0.0
        fDef.filter.groupIndex = -1
        fDef.shape = b2PolygonShape(vertices=vertices)
        hull.CreateFixture(fDef)
    
    def createHull(self,vertices,density):
        hullAttrs = b2BodyDef()
        hullAttrs.position.Set(beginningPos.x,beginningPos.y) 
        hullAttrs.type = b2_dynamicBody;	
        carHull = self.world.CreateBody(hullAttrs)	
        for i in range(len(vertices)):
            self.hullGen(carHull, vertices[i],vertices[(i+1)%8], density)	
        carHull.vertices = vertices	
        return carHull
    
    def getHull(self):
         return self.hull
    
    def getWheels(self):
         return self.wheels

class Road:
    global seed
    def __init__(self,world):
        self.world = world
    
    def createRoad(self):
        global seed
        maxRoad = 200
        endOfRoad = None
        roadPos = b2Vec2(-1,0)
        road = []
        print("SEED:", seed)
        random.seed(seed)
        for k in range(maxRoad):
            #angles generated here are very small numbers <1, these get put into appropriate cos sin equations to accurately calculate angles that road pieces should be rotatated at
            endOfRoad = self.createRoadPiece(roadPos, (random.random()*3 - 1.5) * 1.2*k/maxRoad)
            road.append(endOfRoad)
            endPiece = endOfRoad.fixtures
            if endPiece[0].shape.vertices[3]==b2Vec2(0,0):
                roadCoords = endOfRoad.GetWorldPoint(endPiece[0].shape.vertices[0])
            else:
                roadCoords = endOfRoad.GetWorldPoint(endPiece[0].shape.vertices[3])
            roadPos = roadCoords
        return road
    
    def createRoadPiece(self, p, a):
        global wheelRadMax,wheelRadMin,wheelDenMax,wheelDenMin,hullCoordMax,hullCoordMin,hullDenMax,hullDenMin,roadWidth,roadHeight,beginningPos,carLifePoints,carAccel,isRunning,grav
        road = b2BodyDef()
        fRoad = b2FixtureDef()
        road.position = p
        roadPiece = self.world.CreateBody(road)
        fRoad.friction = 0.5
        fRoad.shape = b2PolygonShape()
        track = []
        track.append(b2Vec2(0,roadHeight))
        track.append(b2Vec2(roadWidth,roadHeight))
        track.append(b2Vec2(0,0))
        track.append(b2Vec2(roadWidth,0))
        cd = self.rotateRoad(track, a)
        fRoad.shape = b2PolygonShape(vertices=cd)
        roadPiece.CreateFixture(fRoad)
        return roadPiece
    
    def rotateRoad(self, c, a):
        newC = []
        for k in range(len(c)):
            nCoords = b2Vec2(0,0)
            nCoords.x = math.cos(a)*(c[k].x) - math.sin(a)*(c[k].y)
            nCoords.y = math.sin(a)*(c[k].x) + math.cos(a)*(c[k].y)
            newC.append(nCoords)
        return newC

class carStuff:
    global beginningPos, carLifePoints, maxDist, genMaxDist, carsEliminated, localGenCounter, genMaxDistCopy, maxDist1, storedDist1, window, gcCopy
    
    def __init__(self,hull,wheels,cDef,xy = [0,0],vel = 0):
        self.xy = xy
        self.vel = vel
        self.hp = carLifePoints
        self.isAlive = False
        self.hull = hull
        self.wheels = wheels
        self.distance = 0
        self.cDef = cDef
    
    def kill(self):
        self.hp = 0
        self.isAlive = True
    
    def getHP(self):
        return self.hp
    
    def isAlive(self):
        return self.isAlive
    
    def healthDown(self):
        self.hp -= 2
    
    def getVelocity(self):
        return self.vel
    
    def getX(self):
        return self.xy[0]
    
    def getY(self):
        return self.xy[1]
    
    def getXY(self):
        return self.xy
    
    def setPosVel(self,p,v):
        if not self.isAlive:
            self.xy = p
            self.vel = v
            self.checkHealth()
            self.checkDist()
            
    def setMaxDist(self,dist):
        self.maxDist = dist
        
    def getMaxDist(self):
        return self.maxDist
    
    def checkHealth(self):
        if self.vel < 0.1:
            self.healthDown()
            if self.hp <= 0:
                self.kill()
    
    def checkDist(self):
        global maxDist, genMaxDist, carsEliminated, distTracker, localGenCounter, genMaxDistCopy, maxDist1, storedDist1, window, gcCopy
        self.distance = self.xy[0]-beginningPos.x
        #storedDist1 here, make it global, refer to it later
        #storedDist1 = self.distance
        #myfontCopy = pygame.font.SysFont("monospace", 16)
        if self.isAlive:
            carsEliminated += 1
            storedDist = self.distance
            storedDist1 = self.distance
            distTracker = self.distance
            print('Distance:', self.distance)
            if (maxDist < storedDist):
                maxDist = storedDist
                #maxDist1 = storedDist1
                
                # rect = pygame.Rect((200, 450), (280, (800/3)))
                # sub = window.subsurface(rect)
                # screenshot = pygame.Surface((280, (800/3)))
                # screenshot.blit(sub, (0,0))
                # pygame.image.save(screenshot, "BestAllTimeSolution.jpg")
               
            if (genMaxDist < storedDist):
                genMaxDist = storedDist
                
                rect1 = pygame.Rect((200, 450), (280, (800/3)))
                sub1 = window.subsurface(rect1)
                screenshot1 = pygame.Surface((280, (800/3)))
                screenshot1.blit(sub1, (0,0))
                pygame.image.save(screenshot1, "BestGenSolution.jpg")
            
            # label40 = myfont9.render(str(round(genMaxDist)), 1, (255,255,0))
            # window.blit(label40, (800, 10))
            
            print('Max Generational Distance :' , genMaxDist)
            genMaxDistCopy = genMaxDist
            if carsEliminated == 20:
                carsEliminated = 0
                genMaxDist = 0
                #localGenCounter += 1
                #y_axis.append(genMaxDist) 
            
            # label41 = myfont9.render(str(round(maxDist)), 1, (255,255,0))
            # window.blit(label41, (800, 285))
            print('Max All Time Distance :' , maxDist)
        
        # plot results
        #plt.ylim(0.0, 200.0)
        #plt.plot(x_axis, y_axis, marker = '.', color = 'b')
        #plt.xlabel('Generation')
        #plt.xticks(np.arange(0,20,step=1))
        #plt.ylabel('Distance Travelled (in meters)')
        #ax.grid(axis='x')
        #plt.show()

class runProgram():
    global wheelRadMax,wheelRadMin,wheelDenMax,wheelDenMin,hullCoordMax,hullCoordMin,hullDenMax,hullDenMin,roadWidth,roadHeight,beginningPos,carLifePoints,carAccel,isRunning,grav,maxDist, genMaxDist, carsEliminated, distTracker, localGenCounter,genMaxDistCopy, maxDist1, storedDist1,gcCopy, window, mutationRate

    def __init__(self):
        self.world = b2World(grav=(0, -9.807), isRunning=True)
        self.popTotal = 20
        self.dead = 0
        self.genCounter = 1
        track = Road(self.world)
        self.terrain = track.createRoad()
        self.pop = []
        self.popInfo = []
        self.createGen()
        self.topPos = [0,0]
        self.top = self.pop[0][0]
        self.draw()
    
    def draw(self):
        global maxDist, genMaxDist, carsEliminated, distTracker, x_axis, y_axis, genMaxDistCopy, y_axis_2, maxDist1, storedDist1, window, gcCopy, mutationRate
        pixels=30.0
        fps=60
        ts=1.0/fps
        displayWidth = 1080
        displayHeight = 800
        playing = True
        #window = pygame.display.set_mode((displayWidth,displayHeight), 0, 32)
        pygame.display.set_caption('Simulating Evolving Cars')
        time = pygame.time.Clock()
        shaders = {staticBody  : (142,142,142,255), dynamicBody : (244,241,144,255)}
        topPos = self.topPos
        x_axis = []
        y_axis = []
        y_axis_2 = []
        
        def drawCircle(circle, circ, f):
            global wheelDenMin, window
            pos=circ.transform*circle.pos*pixels
            yConst = ((self.top.worldCenter.y)*70)
            offsetConst = 300
            if yConst < -offsetConst:
                yConst = -offsetConst
            if yConst > offsetConst:
                yConst = offsetConst
            surfConst, maxedColor, alpha = 50, 255, 100
            circleR, circleG, circleB = 51, 171, 249
            axisX, axisY = 10.0, 25.0
            pos=(pos[0]-self.top.worldCenter.x*30+350, displayHeight-pos[1]+yConst*0.5-200)
            surfaceMidpoint = [int(circle.radius*pixels),int(circle.radius*pixels)]
            background = pygame.Surface((surfConst,surfConst))
            background.set_colorkey((maxedColor,maxedColor,maxedColor))
            background.set_alpha(alpha)
            background.fill((maxedColor,maxedColor,maxedColor)) 
            pygame.draw.circle(background, (circleR, circleG, circleB), surfaceMidpoint, int(circle.radius*pixels),0)
            transformation = circ.transform
            scalar = b2Mul(transformation.q, b2Vec2(axisX,axisY))
            pygame.draw.aaline(background, (maxedColor,0,0), surfaceMidpoint, (surfaceMidpoint[0] -circle.radius*scalar[0], surfaceMidpoint[1] +circle.radius*scalar[1]) )
            window.blit(background, (pos[0]-int(circle.radius*pixels),pos[1]-int(circle.radius*pixels)))
        b2CircleShape.draw=drawCircle
        
        def drawPolygon(p, b, f):
            global window
            yConst = ((self.top.worldCenter.y)*70)
            offsetConst = 300
            if yConst < -offsetConst:
                yConst = -offsetConst
            if yConst > offsetConst:
                yConst = offsetConst
            scalingFactor1, scalingFactor2, sf3, sf4 = 30, 350, 0.5, 200
            vArray=[(b.transform*v)*pixels for v in p.vertices]
            vArray=[(v[0]-self.top.worldCenter.x*scalingFactor1+scalingFactor2, displayHeight-v[1]+yConst*sf3-sf4) for v in vArray]
            pygame.draw.polygon(window, shaders[b.type], vArray)
        polygonShape.draw=drawPolygon
        
        def paused(pause):
            global maxDist, genMaxDist, carsEliminated, distTracker, genMaxDistCopy, window
            myfont1 = pygame.font.SysFont("monospace", 80)
            label11 = myfont1.render("PAUSED", 1, (255,255,0))
            window.blit(label11, ((375), (350)))

            while pause:
                for event in pygame.event.get():

                    if event.type == pygame.QUIT:
                        playing=False
                        pygame.display.quit()
                        pygame.quit()
                        sys.exit()
                    if event.type==pygame.KEYDOWN:
                        if event.key==pygame.K_r:
                            pause = False
                            maxDist = 0
                            genMaxDist = 0
                            carsEliminated = 0
                            distTracker = 0
                            localGenCounter = 0
                            genMaxDistCopy = 0
                            main()
                        if event.key==pygame.K_u:
                            pause = False
                        if event.key == K_ESCAPE:
                            playing=False
                            pygame.display.quit()
                            pygame.quit()
                            sys.exit()
            
                pygame.display.update()
                #clock.tick(15)
        
        def displayImage(di):
            global window, gcCopy
            myfont6 = pygame.font.SysFont("monospace", 16)
            
            bestGenSolImage = pygame.image.load("BestGenSolution.jpg")
            window.blit(bestGenSolImage, (800, 20))
            label12 = myfont6.render("Best Generational Solution:", 1, (255,255,0))
            window.blit(label12, (800, 0))
            
            bestAllTimeImage = pygame.image.load("BestAllTimeSolution.jpg")
            window.blit(bestAllTimeImage, (800, 275))
            label12 = myfont6.render("Best All Time Solution:", 1, (255,255,0))
            window.blit(label12, (800, 285))
            
            label43 = myfont9.render("Distance:", 1, (255,255,0))
            window.blit(label43, (800, 25))
            label40 = myfont9.render(str(round(genMaxDist)), 1, (255,255,0))
            window.blit(label40, (880, 25))
            
            label44 = myfont9.render("Distance:", 1, (255,255,0))
            window.blit(label44, (800, 310))
            label41 = myfont9.render(str(round(maxDist)), 1, (255,255,0))
            window.blit(label41, (880, 310))
            label45 = myfont9.render("Generation:", 1, (255,255,0))
            window.blit(label45, (930, 310))
            label42 = myfont9.render(str(gcCopy), 1, (255,255,0))
            window.blit(label42, (1030, 310))
            
            while di:
                for event in pygame.event.get():
                    if event.type==pygame.KEYDOWN:
                        if event.key==pygame.K_h:
                            di = False
                        if event.key == K_ESCAPE:
                            playing=False
                            pygame.display.quit()
                            pygame.quit()
                            sys.exit()
            
                pygame.display.update()
                
        def showDocumentation(sd):
            global window
            window.fill((0,0,0))
            myfont4 = pygame.font.SysFont("monospace", 16)
            myfont5 = pygame.font.SysFont("monospace", 30)
            label15 = myfont.render("Controls:", 1, (255,255,0))
            window.blit(label15, (0, 0))
            label16 = myfont4.render("Press 'p'/'u' to pause/unpause", 1, (255,255,0))
            window.blit(label16, (0, 20))
            label17 = myfont4.render("Press 'b'/'h' to show/hide the optimal solutions (DON'T PRESS UNTIL AFTER GENERATION 1)", 1, (255,255,0))
            window.blit(label17, (0, 40))
            label18 = myfont4.render("Press 'r' to restart the simulation", 1, (255,255,0))
            window.blit(label18, (0, 60))
            label19 = myfont4.render("Press 'Esc' to close the simulation", 1, (255,255,0))
            window.blit(label19, (0, 80))
            label48 = myfont4.render("Press '1/2/3/4' to change the mutation rate", 1, (255,255,0))
            window.blit(label48, (0, 100))
            label20 = myfont.render("The general idea:", 1, (255,255,0))
            window.blit(label20, (0, 120))
            label21 = myfont4.render("The simulation uses a genetic algorithm to evolve simple 2D cars to travel further along a fixed track.", 1, (255,255,0))
            window.blit(label21, (0, 140))
            label22 = myfont.render("My Approach:", 1, (255,255,0))
            window.blit(label22, (0, 180))
            label23 = myfont4.render("Started with 20 cars each with a randomly generated genome.", 1, (255,255,0))
            window.blit(label23, (0, 200))
            label24 = myfont4.render("The genome controls attributes such as hull shape/density and wheel radius/density.", 1, (255,255,0))
            window.blit(label24, (0, 220))
            label25 = myfont4.render("The cars compete to see which ones travel the furthest.", 1, (255,255,0))
            window.blit(label25, (0, 240))
            label26 = myfont4.render("Pairs of parent cars are chosen to produce offspring that make up the next generation.", 1, (255,255,0))
            window.blit(label26, (0, 260))
            label27 = myfont.render("Methods used:", 1, (255,255,0))
            window.blit(label27, (0, 300))
            label38 = myfont4.render("Encoded solutions are represented by a chromosome data structure.", 1, (255,255,0))
            window.blit(label38, (0, 320))
            label28 = myfont4.render("Selection, Crossover and Mutation are applied iteratively to chromosomes.", 1, (255,255,0))
            window.blit(label28, (0, 340))
            label29 = myfont4.render("Solutions are then assessed using a fitness function based on distance travelled.", 1, (255,255,0))
            window.blit(label29, (0, 360))
            label30 = myfont.render("Was it a success?", 1, (255,255,0))
            window.blit(label30, (0, 400))
            label31 = myfont4.render("Generally, Yes! Though sometimes we encounter evolutionary dead-ends...", 1, (255,255,0))
            window.blit(label31, (0, 420))
            label32 = myfont4.render("Check out the graphs produced during runtime for a detailed view of the simulation results.", 1, (255,255,0))
            window.blit(label32, (0, 440))
            label33 = myfont.render("Why did I make this in the first place?", 1, (255,255,0))
            window.blit(label33, (0, 480))
            label34 = myfont4.render("To show off genetic algorithms and their potential uses in engineering design.", 1, (255,255,0))
            window.blit(label34, (0, 500))
            label36 = myfont.render("Acknowledgements:", 1, (255,255,0))
            window.blit(label36, (0, 540))
            label35 = myfont4.render("Inspired by: https://rednuht.org/genetic_cars_2/ ,http://boxcar2d.com/ and Pavitra Kumar's Genetic Cars.", 1, (255,255,0))
            window.blit(label35, (0, 560))
            label37 = myfont5.render("Press 'c' to close the info menu", 1, (255,255,0))
            window.blit(label37, (250, 600))
                    
            while sd:
                for event in pygame.event.get():
                    if event.type==pygame.KEYDOWN:
                        if event.key==pygame.K_c:
                            sd = False
                        if event.key == K_ESCAPE:
                            playing=False
                            pygame.display.quit()
                            pygame.quit()
                            sys.exit()
                #window.fill((0,0,0))
                pygame.display.update()
        
        while playing:
            
            #attempting to show an optimal solution
            #if self.dead == self.popTotal - 2:
                # rect = pygame.Rect((200, 450), (280, (displayHeight/3)))
                # sub = window.subsurface(rect)
                # screenshot = pygame.Surface((280, (displayHeight/3)))
                # screenshot.blit(sub, (0,0))
                # pygame.image.save(screenshot, "BestGenSolution.jpg")
                #pygame.image.save(window, "BestSolution.jpg")
                #pygame.image.load("BestSolution.jpg")
            
            if self.dead == self.popTotal:
                
                #di = True
                #displayImage(di)
                
                #bestSolImage = pygame.image.load("BestSolution.jpg")
                #window.blit(bestSolImage, (800, 25))
                
                #pygame.display.update() 
                
                #plotting key information during runtime
                fig, ax = plt.subplots()
                x_axis.append(self.genCounter)
                y_axis_2.append(maxDist)
                y_axis.append(genMaxDistCopy)
                plt.ylim(0.0, 225.0)
                
                plt.plot(x_axis, y_axis, marker = '.', color = 'b', label = "Max Generational Distance")
                plt.plot(x_axis, y_axis_2, marker = '.', color = 'r', label = "Max All Time Distance")
                
                #plt.plot(x_axis, y_axis, color = 'b', label = "Max Generational Distance")
                #plt.plot(x_axis, y_axis_2, color = 'r', label = "Max All Time Distance")
                
                plt.xlabel('Generation')
                
                plt.xticks(np.arange(0,21,step=1))
                
                #plt.xticks(np.arange(0,251,step=20))
                
                plt.ylabel('Distance Travelled (in meters)')
                plt.grid(True)
                #ax.grid(axis='x')
                plt.legend(bbox_to_anchor=(0, -0.18), loc='upper left', borderaxespad=0)
                plt.title("Distance Travelled per Generation")
                plt.savefig("ResultsGraph.png",dpi=300)
                plt.show()
                
                self.nextSet()
                self.genCounter += 1
                
                #plot here?
                #fig, ax = plt.subplots()
                #x_axis = []
                #y_axis = []
                #x_axis.append(self.genCounter)
                #y_axis.append(genMaxDist) 
                #plt.ylim(0.0, 200.0)
                #plt.plot(x_axis, y_axis, marker = '.', color = 'b')
                #plt.xlabel('Generation')
                #plt.xticks(np.arange(0,20,step=1))
                #plt.ylabel('Distance Travelled (in meters)')
                #ax.grid(axis='x')
                #plt.show()
            
            #window.blit(bestSolImage, (800, 0))
            #pygame.display.update() 
            self.checkTop()
            self.checkVehicleInfo()
            
            #bestSolImage = pygame.image.load("BestSolution.jpg")
            #window.blit(bestSolImage, (800, 0))
            #pygame.display.update() 
            
            for action in pygame.event.get():
                if action.type==QUIT:
                    playing=False
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
                if action.type==pygame.KEYDOWN:
                    if action.key==pygame.K_r:
                        maxDist = 0
                        genMaxDist = 0
                        carsEliminated = 0
                        distTracker = 0
                        localGenCounter = 0
                        genMaxDistCopy = 0
                        main()
                    if action.key==pygame.K_p:
                        pause = True
                        paused(pause)
                    if action.key==pygame.K_b:
                        di = True
                        displayImage(di)
                    if action.key==pygame.K_i:
                        sd = True
                        showDocumentation(sd)
                    if action.key == K_ESCAPE:
                        playing=False
                        pygame.display.quit()
                        pygame.quit()
                        sys.exit()
                    #mutation rate modifiers
                    if action.key == K_1:
                        mutationRate = 1
                    if action.key == K_2:
                        mutationRate = 2
                    if action.key == K_3:
                        mutationRate = 3
                    if action.key == K_4:
                        mutationRate = 4

                        
            fillA, fillB, stepConst = 0, 100, 10
            window.fill((fillA,fillA,fillA,fillB))
            
            myfont = pygame.font.SysFont("monospace", 20)
            
            # render text
            label1 = myfont.render("Current Generation:", 1, (255,255,0))
            label2 = myfont.render(str(self.genCounter), 1, (255,255,0))
            window.blit(label1, (100, 100))
            window.blit(label2, (420, 100))
            
            label7 = myfont.render("Car Eliminated:", 1, (255,255,0))
            label8 = myfont.render(str(carsEliminated), 1, (255,255,0))
            window.blit(label7, (100, 120))
            window.blit(label8, (420, 120))
            
            label9 = myfont.render("Distance Travelled:", 1, (255,255,0))
            label10 = myfont.render(str(distTracker), 1, (255,255,0))
            window.blit(label9, (100, 140))
            window.blit(label10, (420, 140))
            
            label3 = myfont.render("Max Generational Distance:", 1, (255,255,0))
            label4 = myfont.render(str(genMaxDist), 1, (255,255,0))
            window.blit(label3, (100, 160))
            window.blit(label4, (420, 160))
            
            label5 = myfont.render("Max All Time Distance:", 1, (255,255,0))
            label6 = myfont.render(str(maxDist), 1, (255,255,0))
            window.blit(label5, (100, 180))
            window.blit(label6, (420, 180))
            
            myfont2 = pygame.font.SysFont("monospace", 30)
            myfont3 = pygame.font.SysFont("monospace", 18)
            myfont9 = pygame.font.SysFont("monospace", 15)
            label13 = myfont2.render("Simulating Evolving Cars - 852298", 1, (255,255,0))
            window.blit(label13, (0, 0))
            label14 = myfont3.render("Press 'i' for more info", 1, (255,255,0))
            window.blit(label14, (0, 30))
            
            label38 = myfont3.render("Seed:", 1, (255,255,0))
            window.blit(label38, (0, 780))
            label39 = myfont3.render(str(seed), 1, (255,255,0))
            window.blit(label39, (100, 780))
            
            if mutationRate == 1:
                mutationRef = "0%"
            elif mutationRate == 2:
                mutationRef = "20%"
            elif mutationRate == 3:
                mutationRef = "100%"
            else:
                mutationRef = "Default"
            
            label46 = myfont3.render("Mutation Rate:", 1, (255,255,0))
            window.blit(label46, (275, 780))
            label47 = myfont3.render(mutationRef, 1, (255,255,0))
            window.blit(label47, (475, 780))
            
            #label12 = myfont.render("Best Solution:", 1, (255,255,0))
            #window.blit(label12, (800, 0))
            
            # plot results (causes massive slowdown when done during runtime)
            #fig, ax = plt.subplots()
            #x_axis = []
            #y_axis = []
            #x_axis.append(self.genCounter)
            #y_axis.append(genMaxDist) 
            #plt.ylim(0.0, 200.0)
            #plt.plot(x_axis, y_axis, marker = '.', color = 'b')
            #plt.xlabel('Generation')
            #plt.xticks(np.arange(0,20,step=1))
            #plt.ylabel('Distance Travelled (in meters)')
            #ax.grid(axis='x')
            #plt.show()
            
            for shapes in self.world.bodies:
                for attachments in shapes.fixtures:
                    attachments.shape.draw(shapes,attachments)
            time.tick(fps)
            pygame.display.flip()
            self.world.Step(ts, stepConst, stepConst)
    
    def checkTop(self):
        global window
        sortedData = sorted(self.popInfo,key = lambda x:x.distance)
        for data in sortedData:
            if not data.isAlive:
                self.top = data.hull
    
    def Step(self, settings):
        global window
        super(runProgram, self).Step(settings)
        self.checkVehicleInfo()
        if self.dead == self.popTotal:
            self.nextSet()
    
    def start(self):
        global window
        while True:
            self.checkVehicleInfo()
            if self.dead == self.popTotal:
                self.nextSet()
    
    def getGenCount():
        return self.genCounter
    
    #checkDist() happens first, then this function
    def checkVehicleInfo(self):
        global maxDist1, storedDist1, window, gcCopy
        for index,cars in enumerate(self.popInfo):
            if not cars.isAlive:
                cars.setPosVel([self.pop[index][0].position.x, self.pop[index][0].position.y], self.pop[index][0].linearVelocity.x)
                if cars.isAlive:
                    #if we want to delay deaths for screenshot purposes, should be done here
                    #time.sleep(0.1)
                    #print("STORED DISTANCE:", storedDist1)
                    #not sure if this is more computationally expensive here, or in checkDist()
                    if (maxDist1 < storedDist1):
                        maxDist1 = storedDist1
                        #print("MAX ALLTIME DISTANCE", maxDist1)
                        #taking a pic before cars are removed from screen
                        rect = pygame.Rect((200, 400), (300, 350))
                        sub = window.subsurface(rect)
                        screenshot = pygame.Surface((300, 350))
                        screenshot.blit(sub, (0,0))
                        pygame.image.save(screenshot, "BestAllTimeSolution.jpg")
                        gcCopy = self.genCounter
                        # rect1 = pygame.Rect((200, 450), (280, (800/3)))
                        # sub1 = window.subsurface(rect1)
                        # screenshot1 = pygame.Surface((280, (800/3)))
                        # screenshot1.blit(sub1, (0,0))
                        # pygame.image.save(screenshot1, "BestGenSolution.jpg")
                        
                        
                    for wheel in self.pop[index][1]:
                        if wheel:
                            self.world.DestroyBody(wheel)
                    self.world.DestroyBody(self.pop[index][0])
                    self.pop[index] = None
                    self.dead += 1
                    print ("Cars Eliminated:", self.dead)
    
    
    def sortDist(self):
        self.popInfo = sorted(self.popInfo,key = lambda x:x.distance)
        self.topPos = [self.popInfo[0].hull.worldCenter.x,self.popInfo[0].hull.worldCenter.y]
        self.top = self.popInfo[0].hull  
    
    def checkDuplicate(self,parent,mate):
        duplicate = False
        if parent in mate or parent[::-1] in mate:
            duplicate = True
        return duplicate
    
    #taking 2 two parents from tourney and applying crossover to make children
    def mate(self,parents):
        global mutationRate
        totalAttr = 15
        attrIndex = 0
        parents = [self.popInfo[parents[0]].cDef,self.popInfo[parents[1]].cDef]
        #simulating crossover
        sp1 = random.randint(0,15)
        sp2 = random.randint(0,15)
        
        #each num represents an option for mutation
        # mutationRate0 = 0
        # mutationRate25 = 1
        # mutationRate50 = 2
        # mutationRate75 = 3
        # mutationRate100 = 4
        # mutationRateDefault = 5
        
        while sp1 == sp2:
            sp2 = random.randint(0,15)
        child = Stats()
        currentParent = 0
        
        #wheel num is excluded from mutation - design choice
        child.setWheelNum = parents[currentParent].getWheelNum()
        attrIndex += 1
        currentParent = self.whichParent(attrIndex,currentParent,sp1,sp2)
        
        def mutateHelper(mutationIndex):
            if mutationIndex == 0:
                child.wheelRad[0] = random.uniform(wheelRadMin, wheelRadMax)
                return child.wheelRad[0]
            elif mutationIndex == 1:
                child.wheelRad[1] = random.uniform(wheelRadMin, wheelRadMax)
                return child.wheelRad[1]
            elif mutationIndex == 2:
                wVert1 = []
                wVert1.append(parents[currentParent].wheelVert[0])
                iLeft1 = [i for i in range(8)]
                iNext1 = int(random.random() * (len(iLeft1)-1))
                #print(iLeft1[iNext1])
                wVert1.append(iLeft1[iNext1])
                #print(wVert1)
                iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                #print(parents[currentParent].wheelVert[0])
                #print("Wheel Vertices:", wVert1)
                #wVert1.append(parents[currentParent].wheelVert[0])
                child.setWheelVert(wVert1)
                return child.getWheelVert()
            elif mutationIndex == 3:
                wVert1 = []
                iLeft1 = [i for i in range(8)]
                iNext1 = int(random.random() * (len(iLeft1)-1))
                #print(iLeft1[iNext1])
                wVert1.append(iLeft1[iNext1])
                #print(wVert1)
                iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                #print(parents[currentParent].wheelVert[0])
                wVert1.append(parents[currentParent].wheelVert[1])
                #print("Wheel Vertices:", wVert1)
                #wVert1.append(parents[currentParent].wheelVert[0])
                child.setWheelVert(wVert1)
                return child.getWheelVert()
            elif mutationIndex == 4:
                child.wheelDen[0] = random.randint(wheelDenMin, wheelDenMax)
                return child.wheelDen[0]
            elif mutationIndex == 5:
                child.wheelDen[1] = random.randint(wheelDenMin, wheelDenMax)
                return child.wheelDen[1]
            elif mutationIndex == 6:
                child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
                return child.vertices[0]
            elif mutationIndex == 7:
                child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
                return child.vertices[1]
            elif mutationIndex == 8:
                child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
                return child.vertices[2]
            elif mutationIndex == 9:
                child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
                return child.vertices[3]
            elif mutationIndex == 10:
                child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
                return child.vertices[4]
            elif mutationIndex == 11:
                child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                return child.vertices[5]
            elif mutationIndex == 12:
                child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
                return child.vertices[6]
            elif mutationIndex == 13:
                child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                return child.vertices[7]
            else:
                child.setHullDen = random.randint(hullDenMin, hullDenMax)
                return child.getHullDen()
            
        def sortChromosome(chromosome, mutationIndex):
            chromosome.remove(mutationIndex)
            return chromosome
        
        #0% - nothing changes
        if mutationRate == 1:
            
            for i in range(child.getWheelNum()):
                child.wheelRad[i] = parents[currentParent].wheelRad[i]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            for i in range(child.getWheelNum()):
                child.wheelVert[i] = parents[currentParent].wheelVert[i]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            for i in range(child.getWheelNum()):
                child.wheelDen[i] = parents[currentParent].wheelDen[i]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            for i in range(len(child.getVertices())):
                child.vertices[i] = parents[currentParent].vertices[i]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            child.setHullDen = parents[currentParent].getHullDen()
            offspringCar = carVehicle(self.world, random = False, cDef = child)
        
            return offspringCar
    
        #20% - 3 out of 15 attrs mutate
        elif mutationRate == 2:
            
            mutationIndex1 = random.randint(0,14)
            mutationIndex2 = random.randint(0,14)
            while mutationIndex1 == mutationIndex2:
                mutationIndex2 = random.randint(0,14)
            mutationIndex3 = random.randint(0,14)
            while mutationIndex1 == mutationIndex3 or mutationIndex2 == mutationIndex3:
                mutationIndex3 = random.randint(0,14)
            
            # print("NEW SET:")
            # print(mutationIndex1)
            # print(mutationIndex2)
            # print(mutationIndex3)
           
            # mutationIndex1 = 1
            # mutationIndex2 = 2
            # mutationIndex3 = 3
            
            # mutationIndex4 = random.randint(0,14)
            # while mutationIndex1 == mutationIndex4 or mutationIndex2 == mutationIndex4 or mutationIndex3 == mutationIndex4:
            #     mutationIndex4 = random.randint(0,14)
            # mutationIndex5 = random.randint(0,14)
            # while mutationIndex1 == mutationIndex5 or mutationIndex2 == mutationIndex5 or mutationIndex3 == mutationIndex5 or mutationIndex4 == mutationIndex5:
            #     mutationIndex5 = random.randint(0,14)   
            # mutationIndex6 = random.randint(0,14)
            # while mutationIndex1 == mutationIndex6 or mutationIndex2 == mutationIndex6 or mutationIndex3 == mutationIndex6 or mutationIndex4 == mutationIndex6 or mutationIndex5 == mutationIndex6:
            #     mutationIndex6 = random.randint(0,14) 
            # mutationIndex7 = random.randint(0,14)
            # while mutationIndex1 == mutationIndex7 or mutationIndex2 == mutationIndex7 or mutationIndex3 == mutationIndex7 or mutationIndex4 == mutationIndex7 or mutationIndex5 == mutationIndex7 or mutationIndex6 == mutationIndex7:
            #     mutationIndex7 = random.randint(0,14)    
            # mutationIndex8 = random.randint(0,14)
            # while mutationIndex1 == mutationIndex8 or mutationIndex2 == mutationIndex8 or mutationIndex3 == mutationIndex8 or mutationIndex4 == mutationIndex8 or mutationIndex5 == mutationIndex8 or mutationIndex6 == mutationIndex8 or mutationIndex7 == mutationIndex8:
            #     mutationIndex8 = random.randint(0,14)      
            
                
            wVert1 = []
            chromosome = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            #mutateHelper(mutationIndex1)
            
            chromosome = sortChromosome(chromosome, mutationIndex1)
            
            if mutationIndex1 == 0:
                child.wheelRad[0] = random.uniform(wheelRadMin, wheelRadMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 1:
                child.wheelRad[1] = random.uniform(wheelRadMin, wheelRadMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            elif mutationIndex1 == 2:
                iLeft1 = [i for i in range(8)]
                iNext1 = int(random.random() * (len(iLeft1)-1))
                #print(iLeft1[iNext1])
                wVert1.append(iLeft1[iNext1])
                #print(wVert1)
                iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                #print(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)

            elif mutationIndex1 == 3:
                iLeft1 = [i for i in range(8)]
                iNext1 = int(random.random() * (len(iLeft1)-1))
                #print(iLeft1[iNext1])
                wVert1.append(iLeft1[iNext1])
                #print(wVert1)
                iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                #print(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)

            elif mutationIndex1 == 4:
                child.wheelDen[0] = random.randint(wheelDenMin, wheelDenMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 5:
                child.wheelDen[1] = random.randint(wheelDenMin, wheelDenMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 6:
                child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 7:
                child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 8:
                child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 9:
                child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 10:
                child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 11:
                child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 12:
                child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex1 == 13:
                child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = random.randint(hullDenMin, hullDenMax)
            
            chromosome = sortChromosome(chromosome, mutationIndex2)
            
            if mutationIndex2 == 0:
                child.wheelRad[0] = random.uniform(wheelRadMin, wheelRadMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 1:
                child.wheelRad[1] = random.uniform(wheelRadMin, wheelRadMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 2:
                iLeft1 = [i for i in range(8)]
                iNext1 = int(random.random() * (len(iLeft1)-1))
                #print(iLeft1[iNext1])
                wVert1.append(iLeft1[iNext1])
                #print(wVert1)
                iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                #print(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)

            elif mutationIndex2 == 3:
               iLeft1 = [i for i in range(8)]
               iNext1 = int(random.random() * (len(iLeft1)-1))
               #print(iLeft1[iNext1])
               wVert1.append(iLeft1[iNext1])
               #print(wVert1)
               iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
               #print(parents[currentParent].wheelVert[0])
               attrIndex += 1
               currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
               if len(wVert1) == 2:
                   child.setWheelVert(wVert1)

            elif mutationIndex2 == 4:
                child.wheelDen[0] = random.randint(wheelDenMin, wheelDenMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 5:
                child.wheelDen[1] = random.randint(wheelDenMin, wheelDenMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 6:
                child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 7:
                child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 8:
                child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 9:
                child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 10:
                child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 11:
                child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 12:
                child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex2 == 13:
                child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = random.randint(hullDenMin, hullDenMax)
            
            chromosome = sortChromosome(chromosome, mutationIndex3)
            
            if mutationIndex3 == 0:
                child.wheelRad[0] = random.uniform(wheelRadMin, wheelRadMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 1:
                child.wheelRad[1] = random.uniform(wheelRadMin, wheelRadMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 2:
                iLeft1 = [i for i in range(8)]
                iNext1 = int(random.random() * (len(iLeft1)-1))
                #print(iLeft1[iNext1])
                wVert1.append(iLeft1[iNext1])
                #print(wVert1)
                iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                #print(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)

            elif mutationIndex3 == 3:
               iLeft1 = [i for i in range(8)]
               iNext1 = int(random.random() * (len(iLeft1)-1))
               #print(iLeft1[iNext1])
               wVert1.append(iLeft1[iNext1])
               #print(wVert1)
               iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
               #print(parents[currentParent].wheelVert[0])
               attrIndex += 1
               currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
               if len(wVert1) == 2:
                   child.setWheelVert(wVert1)

            elif mutationIndex3 == 4:
                child.wheelDen[0] = random.randint(wheelDenMin, wheelDenMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 5:
                child.wheelDen[1] = random.randint(wheelDenMin, wheelDenMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 6:
                child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 7:
                child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 8:
                child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 9:
                child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 10:
                child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 11:
                child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 12:
                child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif mutationIndex3 == 13:
                child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = random.randint(hullDenMin, hullDenMax)
            
            #print(chromosome)
            inheritanceIndex1 = chromosome.pop(0)
            inheritanceIndex2 = chromosome.pop(0)
            inheritanceIndex3 = chromosome.pop(0)
            inheritanceIndex4 = chromosome.pop(0)
            inheritanceIndex5 = chromosome.pop(0)
            inheritanceIndex6 = chromosome.pop(0)
            inheritanceIndex7 = chromosome.pop(0)
            inheritanceIndex8 = chromosome.pop(0)
            inheritanceIndex9 = chromosome.pop(0)
            inheritanceIndex10 = chromosome.pop(0)
            inheritanceIndex11 = chromosome.pop(0)
            inheritanceIndex12 = chromosome.pop(0)

            
            if inheritanceIndex1 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex1 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex1 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex1 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            if inheritanceIndex2 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex2 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex2 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex2 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            if inheritanceIndex3 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex3 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex3 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex3 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            if inheritanceIndex4 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex4 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex4 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex4 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            if inheritanceIndex5 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex5 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex5 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex5 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            if inheritanceIndex6 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex6 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex6 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex6 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            if inheritanceIndex7 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex7 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex7 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex7 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
                     
            
            if inheritanceIndex8 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex8 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex8 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex8 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            
            if inheritanceIndex9 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex9 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex9 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex9 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            
            if inheritanceIndex10 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex10 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex10 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex10 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
                
            
            if inheritanceIndex11 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex11 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex11 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex11 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
                
            
            if inheritanceIndex12 == 0:
                child.wheelRad[0] = parents[currentParent].wheelRad[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 1:
                child.wheelRad[1] = parents[currentParent].wheelRad[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 2:
                wVert1.append(parents[currentParent].wheelVert[0])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex12 == 3:
                wVert1.append(parents[currentParent].wheelVert[1])
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                if len(wVert1) == 2:
                    child.setWheelVert(wVert1)
            elif inheritanceIndex12 == 4:
                child.wheelDen[0] = parents[currentParent].wheelDen[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 5:
                child.wheelDen[1] = parents[currentParent].wheelDen[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 6:
                child.vertices[0]=parents[currentParent].vertices[0]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 7:
                child.vertices[1]=parents[currentParent].vertices[1]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 8:
                child.vertices[2]=parents[currentParent].vertices[2]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 9:
                child.vertices[3]=parents[currentParent].vertices[3]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 10:
                child.vertices[4]=parents[currentParent].vertices[4]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 11:
                child.vertices[5]=parents[currentParent].vertices[5]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 12:
                child.vertices[6]=parents[currentParent].vertices[6]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            elif inheritanceIndex12 == 13:
                child.vertices[7]=parents[currentParent].vertices[7]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            else:
                child.setHullDen = parents[currentParent].getHullDen()
            
            offspringCar = carVehicle(self.world, random = False, cDef = child)
        
            return offspringCar
        
            #mutateHelper(mutationIndex2)
            #attrIndex += 1
            #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #mutateHelper(mutationIndex3)
            #attrIndex += 1
            #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
        # #40% - 6 out of 15 attrs mutate
        # elif mutationRate == 3:
        #     mutationIndex1 = random.randint(0,14)
        #     mutationIndex2 = random.randint(0,14)
        #     mutationIndex3 = random.randint(0,14)
        #     mutationIndex4 = random.randint(0,14)
        #     mutationIndex5 = random.randint(0,14)
        #     mutationIndex6 = random.randint(0,14)
        
        # #60% - 9 out of 15 attrs mutate
        # elif mutationRate == 4:
        #     mutationIndex1 = random.randint(0,14)
        #     mutationIndex2 = random.randint(0,14)
        #     mutationIndex3 = random.randint(0,14)
        #     mutationIndex4 = random.randint(0,14)
        #     mutationIndex5 = random.randint(0,14)
        #     mutationIndex6 = random.randint(0,14)
        #     mutationIndex7 = random.randint(0,14)
        #     mutationIndex8 = random.randint(0,14)
        #     mutationIndex9 = random.randint(0,14)
        
        # #80% - 12 out of 15 attrs mutate
        # elif mutationRate == 5:
        #     mutationIndex1 = random.randint(0,14)
        #     mutationIndex2 = random.randint(0,14)
        #     mutationIndex3 = random.randint(0,14)
        #     mutationIndex4 = random.randint(0,14)
        #     mutationIndex5 = random.randint(0,14)
        #     mutationIndex6 = random.randint(0,14)
        #     mutationIndex7 = random.randint(0,14)
        #     mutationIndex8 = random.randint(0,14)
        #     mutationIndex9 = random.randint(0,14)
        #     mutationIndex10 = random.randint(0,14)
        #     mutationIndex11 = random.randint(0,14)
        #     mutationIndex12 = random.randint(0,14)
        
        #100% - all 15 attrs change
        elif mutationRate == 3:
            for i in range(child.getWheelNum()):
                child.wheelRad[i] = random.uniform(wheelRadMin, wheelRadMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            wVert1 = []
            iLeft1 = [i for i in range(8)]
            for i in range(child.getWheelNum()):
                iNext1 = int(random.random() * (len(iLeft1)-1))
                #print(iLeft1[iNext1])
                wVert1.append(iLeft1[iNext1])
                #print(wVert1)
                iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #print("Wheel Vertices:", wVert1)
            #wVert1.append(parents[currentParent].wheelVert[0])
            child.setWheelVert(wVert1)
            
            for i in range(child.getWheelNum()):
                child.wheelDen[i] = random.randint(wheelDenMin, wheelDenMax)
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            #currently makes a brand new shape
            child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
            attrIndex += 1
            currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            child.setHullDen = random.randint(hullDenMin, hullDenMax)
            offspringCar = carVehicle(self.world, random = False, cDef = child)
            
            return offspringCar
        
        #Default - sections of genome are mutatated randomly
        else:
            #mutation cases covered with crossover simultaneously using index to ref parts to be mutated
            mutationIndex = random.randint(0,14)
        
            # while sp1 == sp2:
                #     sp2 = random.randint(0,15)
                # child = Stats()
                # currentParent = 0
                # child.setWheelNum = parents[currentParent].getWheelNum()
                # attrIndex += 1
                # currentParent = self.whichParent(attrIndex,currentParent,sp1,sp2)
        
            #mutation for wheel radius
            if mutationIndex <= 2:
                for i in range(child.getWheelNum()):
                    child.wheelRad[i] = random.uniform(wheelRadMin, wheelRadMax)
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelVert[i] = parents[currentParent].wheelVert[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelDen[i] = parents[currentParent].wheelDen[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(len(child.getVertices())):
                    child.vertices[i] = parents[currentParent].vertices[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.setHullDen = parents[currentParent].getHullDen()
                offspringCar = carVehicle(self.world, random = False, cDef = child)
                #return offspringCar
        
            #mutation for wheel vertices
            elif mutationIndex > 2 and mutationIndex <= 5:
                for i in range(child.getWheelNum()):
                    child.wheelRad[i] = parents[currentParent].wheelRad[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                wVert1 = []
                iLeft1 = [i for i in range(8)]
                for i in range(child.getWheelNum()):
                    iNext1 = int(random.random() * (len(iLeft1)-1))
                    #print(iLeft1[iNext1])
                    wVert1.append(iLeft1[iNext1])
                    #print(wVert1)
                    iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                #print("Wheel Vertices:", wVert1)
                #wVert1.append(parents[currentParent].wheelVert[0])
                child.setWheelVert(wVert1)
            
            
                #for mutating a single vertex only, 1st then 2nd
                # wVert1 = []
                # wVert1.append(parents[currentParent].wheelVert[0])
                # attrIndex += 1
                # currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                # iLeft1 = [i for i in range(8)]
                # iNext1 = int(random.random() * (len(iLeft1)-1))
                # #print(iLeft1[iNext1])
                # wVert1.append(iLeft1[iNext1])
                # #print(wVert1)
                # iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                # #print(parents[currentParent].wheelVert[0])
                # attrIndex += 1
                # currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                # #print("Wheel Vertices:", wVert1)
                # #wVert1.append(parents[currentParent].wheelVert[0])
                # child.setWheelVert(wVert1)
            
            
                # wVert1 = []
                # iLeft1 = [i for i in range(8)]
                # iNext1 = int(random.random() * (len(iLeft1)-1))
                # #print(iLeft1[iNext1])
                # wVert1.append(iLeft1[iNext1])
                # #print(wVert1)
                # iLeft1 = iLeft1[:iNext1] + iLeft1[iNext1+1:]
                # #print(parents[currentParent].wheelVert[0])
                # attrIndex += 1
                # currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                # wVert1.append(parents[currentParent].wheelVert[1])
                # attrIndex += 1
                # currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                # #print("Wheel Vertices:", wVert1)
                # #wVert1.append(parents[currentParent].wheelVert[0])
                # child.setWheelVert(wVert1)
            
            
                #enforcing that mutated wheels should generally appear on the vertices of the hull bottoms (vertices 6-8) - alt method
                # vertOne = random.randint(6,8)
                # vertTwo = random.randint(6,8)
                # while vertOne == vertTwo:
                    #     vertTwo = random.randint(6,8)
                # child.wheelVert[0] = vertOne
                # attrIndex += 1
                # currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                # child.wheelVert[1] = vertTwo
                # attrIndex += 1
                # currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
            
                for i in range(child.getWheelNum()):
                    child.wheelDen[i] = parents[currentParent].wheelDen[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(len(child.getVertices())):
                    child.vertices[i] = parents[currentParent].vertices[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.setHullDen = parents[currentParent].getHullDen()
                offspringCar = carVehicle(self.world, random = False, cDef = child)
            
            #mutation for wheel density
            elif mutationIndex > 5 and mutationIndex <= 8:
                for i in range(child.getWheelNum()):
                    child.wheelRad[i] = parents[currentParent].wheelRad[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelVert[i] = parents[currentParent].wheelVert[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelDen[i] = random.randint(wheelDenMin, wheelDenMax)
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(len(child.getVertices())):
                    child.vertices[i] = parents[currentParent].vertices[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.setHullDen = parents[currentParent].getHullDen()
                offspringCar = carVehicle(self.world, random = False, cDef = child)
                #return offspringCar
        
            # #mutation for hull density
            # else:
            #     for i in range(child.getWheelNum()):
            #         child.wheelRad[i] = parents[currentParent].wheelRad[i]
            #         attrIndex += 1
            #         currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #     for i in range(child.getWheelNum()):
            #         child.wheelVert[i] = parents[currentParent].wheelVert[i]
            #         attrIndex += 1
            #         currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #     for i in range(child.getWheelNum()):
            #         child.wheelDen[i] = parents[currentParent].wheelDen[i]
            #         attrIndex += 1
            #         currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #     for i in range(len(child.getVertices())):
            #         child.vertices[i] = parents[currentParent].vertices[i]
            #         attrIndex += 1
            #         currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #     child.setHullDen = random.randint(hullDenMin, hullDenMax)
            #     offspringCar = carVehicle(self.world, random = False, cDef = child)
        
            #mutation for car shape
            elif mutationIndex > 8 and mutationIndex <= 11:
                for i in range(child.getWheelNum()):
                    child.wheelRad[i] = parents[currentParent].wheelRad[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelVert[i] = parents[currentParent].wheelVert[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelDen[i] = parents[currentParent].wheelDen[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                #currently makes a brand new shape
                child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                attrIndex += 1
                currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                #what if we split it 50/50 with an index to choose vertices to randomize, alt method for mutation - not the best, you end up with cars with shapes that don't work well with the other attributes
                #vertexIndex = round(random.random())
                #random vertices for 0,2,4,6
                #if vertexIndex == 0:
                    #child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[1]=parents[currentParent].vertices[1]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[3]=parents[currentParent].vertices[3]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[5]=parents[currentParent].vertices[5]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[7]=parents[currentParent].vertices[7]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                #random vertices for 1,3,5,7
                #else:
                    #child.vertices[0]=parents[currentParent].vertices[0]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[2]=parents[currentParent].vertices[2]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[4]=parents[currentParent].vertices[4]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[6]=parents[currentParent].vertices[6]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                #attempting to control the alt method better wiht just one vertex being mutated at a time - not the best practise
                #vertexIndex = random.randint(0, 7)
            
                #if vertexIndex == 0:
                    #child.vertices[0]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,0))
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[1]=parents[currentParent].vertices[1]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[2]=parents[currentParent].vertices[2]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[3]=parents[currentParent].vertices[3]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[4]=parents[currentParent].vertices[4]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[5]=parents[currentParent].vertices[5]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[6]=parents[currentParent].vertices[6]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #child.vertices[7]=parents[currentParent].vertices[7]
                    #attrIndex += 1
                    #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                #elif vertexIndex == 1:
                    #     child.vertices[0]=parents[currentParent].vertices[0]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[1]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin, random.random()*hullCoordMax + hullCoordMin  ))
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[2]=parents[currentParent].vertices[2]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[3]=parents[currentParent].vertices[3]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[4]=parents[currentParent].vertices[4]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[5]=parents[currentParent].vertices[5]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[6]=parents[currentParent].vertices[6]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[7]=parents[currentParent].vertices[7]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                #elif vertexIndex == 2:
                    #     child.vertices[0]=parents[currentParent].vertices[0]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[1]=parents[currentParent].vertices[1]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[2]=(b2Vec2(0, random.random()*hullCoordMax + hullCoordMin))
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[3]=parents[currentParent].vertices[3]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[4]=parents[currentParent].vertices[4]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[5]=parents[currentParent].vertices[5]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[6]=parents[currentParent].vertices[6]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[7]=parents[currentParent].vertices[7]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                
                #elif vertexIndex == 3:
                    #     child.vertices[0]=parents[currentParent].vertices[0]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[1]=parents[currentParent].vertices[1]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[2]=parents[currentParent].vertices[2]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[3]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin, random.random()*hullCoordMax + hullCoordMin))
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[4]=parents[currentParent].vertices[4]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[5]=parents[currentParent].vertices[5]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[6]=parents[currentParent].vertices[6]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[7]=parents[currentParent].vertices[7]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                
                #elif vertexIndex == 4:
                    #     child.vertices[0]=parents[currentParent].vertices[0]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[1]=parents[currentParent].vertices[1]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[2]=parents[currentParent].vertices[2]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[3]=parents[currentParent].vertices[3]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[4]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,0))
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[5]=parents[currentParent].vertices[5]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[6]=parents[currentParent].vertices[6]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[7]=parents[currentParent].vertices[7]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                
                #elif vertexIndex == 5:
                    #     child.vertices[0]=parents[currentParent].vertices[0]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[1]=parents[currentParent].vertices[1]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[2]=parents[currentParent].vertices[2]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[3]=parents[currentParent].vertices[3]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[4]=parents[currentParent].vertices[4]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[5]=(b2Vec2(-random.random()*hullCoordMax - hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[6]=parents[currentParent].vertices[6]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[7]=parents[currentParent].vertices[7]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                
                #elif vertexIndex == 6:
                    #     child.vertices[0]=parents[currentParent].vertices[0]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[1]=parents[currentParent].vertices[1]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[2]=parents[currentParent].vertices[2]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[3]=parents[currentParent].vertices[3]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[4]=parents[currentParent].vertices[4]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[5]=parents[currentParent].vertices[5]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[6]=(b2Vec2(0,-random.random()*hullCoordMax - hullCoordMin))
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[7]=parents[currentParent].vertices[7]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                
                #else:
                    #     child.vertices[0]=parents[currentParent].vertices[0]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[1]=parents[currentParent].vertices[1]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[2]=parents[currentParent].vertices[2]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[3]=parents[currentParent].vertices[3]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[4]=parents[currentParent].vertices[4]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[5]=parents[currentParent].vertices[5]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[6]=parents[currentParent].vertices[6]
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                    #     child.vertices[7]=(b2Vec2(random.random()*hullCoordMax + hullCoordMin,-random.random()*hullCoordMax - hullCoordMin))
                    #     attrIndex += 1
                    #     currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            
                child.setHullDen = parents[currentParent].getHullDen()
                offspringCar = carVehicle(self.world, random = False, cDef = child)
            
            #mutation for hull density
            else:
                for i in range(child.getWheelNum()):
                    child.wheelRad[i] = parents[currentParent].wheelRad[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelVert[i] = parents[currentParent].wheelVert[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(child.getWheelNum()):
                    child.wheelDen[i] = parents[currentParent].wheelDen[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                for i in range(len(child.getVertices())):
                    child.vertices[i] = parents[currentParent].vertices[i]
                    attrIndex += 1
                    currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
                child.setHullDen = random.randint(hullDenMin, hullDenMax)
                offspringCar = carVehicle(self.world, random = False, cDef = child)
            
            return offspringCar
        
            #for i in range(child.getWheelNum()):
                #child.wheelRad[i] = parents[currentParent].wheelRad[i]
                #attrIndex += 1
                #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #for i in range(child.getWheelNum()):
                #child.wheelVert[i] = parents[currentParent].wheelVert[i]
                #attrIndex += 1
                #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #for i in range(child.getWheelNum()):
                #child.wheelDen[i] = parents[currentParent].wheelDen[i]
                #attrIndex += 1
                #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #for i in range(len(child.getVertices())):
                #child.vertices[i] = parents[currentParent].vertices[i]
                #attrIndex += 1
                #currentParent = self.whichParent(attrIndex, currentParent, sp1, sp2)
            #child.setHullDen = parents[currentParent].getHullDen()
            #offspringCar = carVehicle(self.world, random = False, cDef = child)
            #return offspringCar
    
    #switches parent index
    def whichParent(self,index,lp, sp1, sp2):
        if index == sp1 or index == sp2:
            if lp == 0:
                return 1
            else:
                return 0
        else:
            return lp
    
    #prototype mutation operator
    #def mutate(self,child):
        #mutationIndex = random.randint(0,15) 
        #if mutationIndex <= 5:
            #for i in range(child.getWheelNum()):
                #child.wheelRad[i] = random.uniform(wheelRadMin, wheelRadMax)
        #elif mutationIndex > 3 and mutationIndex <= 6:
            #for i in range(child.getWheelNum()):
                #child.wheelVert[i] = 
        #elif mutationIndex > 5 and mutationIndex <= 10:
            #for i in range(child.getWheelNum()):
                #child.wheelDen[i] = random.randint(wheelDenMin, wheelDenMax)
        #elif mutationIndex > 9 and mutationIndex <= 12:
            #for i in range(len(child.getVertices())):
                #child.vertices[i] = 
        #elif mutationIndex > 10 and mutationIndex <= 15:
            #child.setHullDen = random.randint(hullDenMin, hullDenMax)
        
        #return child
                
    #tournament selection done here
    def getParent(self, mate):
        mating_pool = []
        for i in range(self.popTotal):
        #for i in range(10):
            randCar = random.randint(0, self.popTotal-1)
            while randCar == i:
                randCar = random.randint(0, self.popTotal-1)
            carA_Dist = self.popInfo[i].distance
            carB_Dist = self.popInfo[randCar].distance
            if carA_Dist > carB_Dist:
                mating_pool.append(i)
            else:
                mating_pool.append(randCar)
        
        p1 = random.randint(0, (len(mating_pool)-1))
        p2 = random.randint(0, (len(mating_pool)-1))
        while p2 == p1 and not self.checkDuplicate([p1,p2], mate):
            p1 = random.randint(0, (len(mating_pool)-1))
            p2 = random.randint(0, (len(mating_pool)-1))
        return [p1,p2]
    
    def nextSet(self):
        self.sortDist()
        #elitism enforcement
        #n = 1
        n = 3 #3 elites
        newPop = []
        newPopInfo = []
        for i in range(n):
            newCar = carVehicle(self.world,random=False,cDef = self.popInfo[i].cDef)
            newPop.append([newCar.hull,newCar.wheels])
            newPopInfo.append(carStuff(self.popInfo[i].hull,self.popInfo[i].wheels,self.popInfo[i].cDef))
        mates = []
        #tournament selection next
        while len(newPop) < self.popTotal:
            parents = self.getParent(mates)
            mates.append(parents)
            child = self.mate(parents)
            #mutation
            #mutated_child = self.mutate(child)
            newPop.append([child.hull,child.wheels])
            newPopInfo.append(carStuff(child.hull,child.wheels,child.cDef))
        print ("\n")
        gc = self.genCounter + 1
        print ("GENERATION", gc, ":")
        self.dead = 0
        for index,elem in enumerate(newPopInfo):
            self.popInfo[index] = elem
        for index,elem in enumerate(newPop):
            self.pop[index] = elem
        self.sortDist()
    
    def createGen(self):
        for i in range(self.popTotal):
            t = carVehicle(self.world)
            self.pop.append([t.getHull(), t.getWheels()])
            self.popInfo.append(carStuff(t.getHull(), t.getWheels(), t.cDef))

def main():
    runProgram()

main = runProgram()
