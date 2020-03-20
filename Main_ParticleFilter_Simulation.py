import WorldCs as worldCs
import IdealRobotCs as idelRob
import RobotCs as Robot
import MapCs as Map
import LandmarkCs as Landmark
import IdealCameraCs as Idealcam
import CameraCs as Camera
import WorldCs as World
import EstimationAgentCS as EstAgent
import MonteCarloLocalizationCs as MCL

import math
import numpy as np

timespan = 200
timeinterval = 1
world = World.World(timespan,timeinterval)


#Map
map = Map.Map()
for pos in [(-4,-5),(-2.5,-2.5),(4,4)]:
    lm = Landmark.Landmark(pos[0],pos[1])
    map.append_landmark( lm )
world.append(map)

#Agent
start_pos=np.array([0,0,0]).T
particlsNum = 100
estimator = MCL.MonteCarloLocalization(start_pos,particlsNum,envmap=map)
circring = EstAgent.EstimationAgent(nu = 0.4,omega=10.0/180.0*math.pi,estimator = estimator,time_interval = timeinterval)


#Robot
robot = Robot.Robot(np.array([0,0,0]).T,sensor=Camera.Camera(map,timeinterval),agent=circring
    ,randam_biasRatio=(0.1,0.1),has_obsNoise=True,has_kidnap=False)


world.append(robot)

#Start the Animation !  
world.draw()
