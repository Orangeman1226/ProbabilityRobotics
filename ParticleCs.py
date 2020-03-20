from scipy.stats import expon , norm ,uniform ,multivariate_normal
import math
import matplotlib.patches as patches
import numpy as np
from enum import Enum 
import IdealRobotCs as IRobot
import IdealCameraCs as ICamera



class Particle():
    def __init__(self,ini_pose,weight):
        self.pose = ini_pose
        self.weight = weight


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
    
    def observeation_update(self,observeation,envmap,distance_sigma_forLikelihood,direction_sigma_forLikelihood):
        
        #一回で観測されたランドマークの観測結果それぞれ取り出す
        #観測値にはノイズや外乱が載っている値となっている
        for obs in observeation:
            if obs[0] is None and obs[1] is None : continue
            else:

                obs_pos = obs[0]#現在のロボット位置からノイズや外乱が載っているであろう観測されたランドマーク
                            #までの実測値（距離:r と 方向:phi）を算出
                obs_id = obs[1] #観測されたランドマークの地図上のID
        
                #観測されたランドマークの地図上のIDより、
                #本来、観測されるはずの位置座標を取得
                landmark_pos = envmap.landmarks[obs_id].pos
                #粒子のそれぞれの位置とランドマークの位置:r・方向:phiを算出
                landmark_polar = ICamera.IdealCamera.observeation_function(self.pose,landmark_pos)
            
                #距離が遠くなるに連れて、尤度の確率分布（２次元ガウス分布）の距離のバラツキは比例するとする。
                #観測した距離と方向の尤度の確率分布の共分散行列を算出する
                distance_dev = distance_sigma_forLikelihood * landmark_polar[0]
                covariance = np.diag(np.array([distance_dev**2,direction_sigma_forLikelihood**2]))
                
                #現在のロボット位置　と　理想（ノイズ外乱がない）状態で観測されるはずの位置との距離rと方向phiを
                #平均とし、前述の共分散より定義した尤度関数（２次元ガウス確率密度分布）より、実際に観測した実測値（距離:r と 方向:phi）の
                #確率密度値を取得し、粒子の重さと積をし、粒子の重さ更新する。
                self.weight = self.weight * multivariate_normal(mean = landmark_polar , cov = covariance).pdf(obs_pos) 
