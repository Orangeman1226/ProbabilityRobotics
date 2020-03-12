import RobotCs as Robot
import IdealRobotCs as IRobot
import WorldCs as World
import AgentCs as Agent

import math
import numpy as np

world = World.World(50,0.1)

circling = Agent.Agent(0.7, 10.0/180 * math.pi)

for i in range(30): #Stuck以外の雑音(bias or obsなど)はOFFする。
    stucked_robot = Robot.Robot(np.array([0,0,0]).T,sensor=None,agent=circling,obstacles_per_meter=0
    ,randam_biasRatio=(0.0,0.0),expected_stuck_time=10,expected_escape_time =5,color="red",has_obsNoise=True)
    world.append(stucked_robot)


NoStuck_robot = IRobot.IdealRobot(np.array([0,0,0]).T,sensor=None,agent=circling)
world.append(NoStuck_robot)

world.draw()