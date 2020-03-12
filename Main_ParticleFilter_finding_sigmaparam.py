import WorldCs as worldCs
import RobotCs as Robot
import MapCs as Map
import LandmarkCs as Landmark
import CameraCs as Camera
import WorldCs as World
import AgentCs as ag
import MonteCarloLocalizationCs as MCL

import math
import numpy as np
import pandas as pd
import copy 

timespan = 40
timeinterval = 0.1
world = World.World(timespan,timeinterval)
robots = []

#Map
map = Map.Map()

world.append(map)

#Agent
Ag = ag.Agent(0.1,0.0)

#Robot 
robot = Robot.Robot(np.array([0,0,0]).T,sensor=None,agent=Ag
    ,randam_biasRatio=(0.1,0.1),has_obsNoise=True,has_kidnap=False)

for i in range(100):

    #ドッペルゲンガーロボット生成
    drobot = copy.copy(robot)
    #ドッペルゲンガーの障害物雑音のみドローして更新
    drobot.distance_untilSteppedonObsNoise = drobot.obsNoise_pdf.rvs()
    
    robots.append(drobot)
    world.append(drobot)

world.append(robot)

#Start the Animation !
world.draw()

poses = pd.DataFrame([ [math.sqrt(robot.pose[0]**2+robot.pose[1]**2),robot.pose[2] ] for robot in robots],columns=["r","theta"] )
print(poses.transpose())
print(poses["theta"].var())
print(poses["r"].mean())
print(poses["theta"].var()/poses["r"].mean())


