import RobotCs as Robot
import IdealRobotCs as IRobot
import WorldCs as World
import AgentCs as Agent

import math
import numpy as np

world = World.World(30,0.1)

circling = Agent.Agent(0.2 , 10.0/180 * math.pi)

Nobias_robot = IRobot.IdealRobot(np.array([0,0,0]).T,sensor=None,agent=circling)
world.append(Nobias_robot)

bias_robot = Robot.Robot(np.array([0,0,0]).T,sensor=None,agent=circling,color="red",obstacles_per_meter=0,randam_biasRatio=(0.2,0.2),has_obsNoise=False)
world.append(bias_robot)

world.draw()