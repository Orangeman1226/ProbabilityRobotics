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


map = Map.Map()
map.append_landmark(Landmark.Landmark(1,0))
world.append(map)

distance = []
direction = []

stdArray=[]

for i_std in range(10):
    for i in range(100):
        cam = Camera.Camera(map,timeinterval,occfarOcclusion_prob=0.001)
        d = cam.data(np.array([0,0,0]).T)
        if len(d) > 0:
            if d[0][0] is not None:
                distance.append(d[0][0][0])
                direction.append(d[0][0][1])
    
    df = pd.DataFrame()
    df["distance_data"] = distance
    df["direction_data"] = direction
    stdArray.append( [i_std,df.std()] )
    distance.clear()
    direction.clear()



print(stdArray)
