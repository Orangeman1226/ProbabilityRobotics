import WorldCs as worldCs
import IdealRobotCs as idelRob
import AgentCs as Ag
import MapCs as Map
import LandmarkCs as Landmark
import IdealCameraCs as Idealcam

import math
import numpy as np

#time_span Max : 10
#最大以上を指定するとGifが保存されないので注意！
world= worldCs.World(10,0.2)


#Map
map = Map.Map()
map.append_landmark(Landmark.Landmark(5,2.5))
map.append_landmark(Landmark.Landmark(-6,-7))
map.append_landmark(Landmark.Landmark(8,6))
map.append_landmark(Landmark.Landmark(-4,-4))
world.append(map)

#Agent
straight = Ag.Agent(0.2,0.0)
circring1 = Ag.Agent(0.7,10.0/180.0 *math.pi)
circring2 = Ag.Agent(0.8,10.0/180.0*math.pi)


#Robot
robot1 = idelRob.IdealRobot(np.array([2,1,math.pi/6]).T,sensor=Idealcam.IdealCamera(map),agent=circring1)
robot2 = idelRob.IdealRobot(np.array([0,-2,math.pi/5*6]).T,agent=circring2,sensor=Idealcam.IdealCamera(map),color = "red")
world.append(robot1)
world.append(robot2)



#Start the Animation !
world.draw()

#IdealCamera
comera = Idealcam.IdealCamera(map)                                                                                      
obs = comera.data(robot2.pose)
print(obs)


#print(idelRob.IdealRobot.state_transition(0.1,0.00,1.0,np.array([0,0,0]).T))
#print(idelRob.IdealRobot.state_transition(0.1,10.00/180 * math.pi,9.0,np.array([0,0,0]).T))
#print(idelRob.IdealRobot.state_transition(0.1,10.00/180 * math.pi,18.0,np.array([0,0,0]).T))