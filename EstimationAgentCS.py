import IdealRobotCs as IRobot
import OccNoisemarkCs as Noisemark
from scipy.stats import expon , norm ,uniform
import math
import matplotlib.patches as patches
import numpy as np
from enum import Enum 
import AgentCs as Ag

class EstimationAgent(Ag.Agent):
    def __init__(self,time_interval,nu,omega,estimator):
        super().__init__(nu,omega)
        self.estimator = estimator
        self.time_interval = time_interval
        
        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def decision(self,observeation=None):
        
        #一つ前の制御指令値で粒子の姿勢を更新する
        self.estimator.motion_update(self.prev_nu,self.prev_omega,self.time_interval)
        self.prev_nu,self.prev_omega = self.nu,self.omega

        return self.nu,self.omega

    def draw(self,ax,elems):
       self.estimator.draw(ax,elems)
