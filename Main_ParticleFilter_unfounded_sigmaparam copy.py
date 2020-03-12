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

timespan = 100
timeinterval = 0.5
world = World.World(timespan,timeinterval)


#Map
map = Map.Map()
for pos in [(-4,2),(-2,3),(3,3)]:
    lm = Landmark.Landmark(pos[0],pos[1])
    map.append_landmark( lm )
world.append(map)

#Agent
start_pos=np.array([2,2,math.pi/6]).T
particlsNum = 100
estimator = MCL.MonteCarloLocalization(start_pos,particlsNum,motion_noise_stds={"nn":1,"no":2,"on":3,"oo":4})
circring = EstAgent.EstimationAgent(nu = 0.7,omega=10.0/180.0*math.pi,estimator = estimator,time_interval = timeinterval)

#Robot
robot = Robot.Robot(np.array([0,0,0]).T,sensor=Camera.Camera(map,time_interval=timeinterval),agent=circring
    ,randam_biasRatio=(0.1,0.1),has_obsNoise=True,has_kidnap=True)

world.append(robot)

#Start the Animation !
world.draw()
