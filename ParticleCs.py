from scipy.stats import expon , norm ,uniform
import math
import matplotlib.patches as patches
import numpy as np
from enum import Enum 
import IdealRobotCs as IRobot

class Particle():
    def __init__(self,ini_pose):
        self.pose = ini_pose


    def motion_update(self,nu,omega,delta_time,motionNoise_rate_pdf):
        ns = motionNoise_rate_pdf.rvs() #４次元ガウス分布からドローする
        
        # 制御指令値U=(nu,omega)とすると、粒子の移動U'=(n',o') をもとめると・・・
        #             n=nu,o=omegaとする
        # 　　　
        #      [ n' ] = [n] + [ σ_nn・squrt ( ⊿t / |n|  ) + σ_nn・squrt ( ⊿t / |n|  ) ] 
        #  　  [ o' ] = [o] + [ σ_on・squrt ( ⊿t / |b|  ) + σ_oo・squrt ( ⊿t / |o|  ) ]
        #
        #
        addNoise_nu     = nu    +  ns[0]*math.sqrt(abs(nu)/delta_time) + ns[1]*math.sqrt(abs(omega)/delta_time)  
        addNoise_omega  = omega +  ns[2]*math.sqrt(abs(nu)/delta_time) + ns[3]*math.sqrt(abs(omega)/delta_time)  
    
        self.pose = IRobot.IdealRobot.state_transition(addNoise_nu,addNoise_omega,delta_time,self.pose)
    

        