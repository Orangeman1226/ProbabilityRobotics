import RobotCs as Robot
import WorldCs as World
import AgentCs as Agent

import math
import numpy as np

world = World.World(200,0.5)

for i in range(1):
    circling = Agent.Agent(0.6 , 10.0/180 * math.pi)
    robot = Robot.Robot(np.array([0,0,0]).T,sensor=None,agent=circling,randam_biasRatio=(0.0,0.0),has_obsNoise=True)
    world.append(robot)

world.draw()

