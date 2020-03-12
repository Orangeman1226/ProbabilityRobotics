import WorldCs as worldCs
import IdealRobotCs as idelRob
import RobotCs as Robot
import AgentCs as Ag
import MapCs as Map
import LandmarkCs as Landmark
import IdealCameraCs as Idealcam
import CameraCs as Camera
import WorldCs as World

import math
import numpy as np

timespan = 100
timeinterval = 0.5
world = World.World(timespan,timeinterval)


#Map
map = Map.Map()
map.append_landmark(Landmark.Landmark(5,2.5))
map.append_landmark(Landmark.Landmark(-6,0))
map.append_landmark(Landmark.Landmark(6.2,6))
map.append_landmark(Landmark.Landmark(-4,5))
world.append(map)

#Agent
#straight = Ag.Agent(0.2,0.0)
circring1 = Ag.Agent(0.7,10.0/180.0*math.pi)
#circring2 = Ag.Agent(0.8,10.0/180.0*math.pi)


#Robot
robot1 = Robot.Robot(np.array([0,0,0]).T,sensor=Camera.Camera(map,time_interval=timeinterval),agent=circring1
    ,randam_biasRatio=(0.0,0.0),has_obsNoise=False,has_kidnap=False)

world.append(robot1)

#Start the Animation !
world.draw()

