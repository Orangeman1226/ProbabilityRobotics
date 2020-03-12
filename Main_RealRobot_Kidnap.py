import RobotCs as Robot
import WorldCs as World
import AgentCs as Agent

import math
import numpy as np

world = World.World(30,0.1)

for i in range(1):
    circling = Agent.Agent(0.6 , 10.0/180 * math.pi)
    robot = Robot.Robot(np.array([0,0,0]).T,sensor=None,agent=circling
    ,randam_biasRatio=(0.0,0.0),has_obsNoise=False,has_kidnap=True,expected_kidnap_time= 5)
    world.append(robot)

world.draw()