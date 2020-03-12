import math
import numpy as np
from scipy.stats import expon , norm ,uniform
import IdealCameraCs as ICamera
from enum import Enum 
import OccSensorNoisemarkCs as OccSensNisemark
import WorldCs as wo
from enum import Enum 

class KindofObsEventNoise(Enum):
    noEvent ="-"
    phantom = "p"
    oversight = "o"
    occulusion = "c"
    overRange = "r"
    
#
# センサには、偶然誤差、系統誤差、過失誤差等があり、具体的に言うといかが考えられる
# 
# １）計測値に対する雑音：ガウス性の雑音でセンサ値がばらつく
#    ⇒ガウス分布：計測値を平均に、距離に対しては計測値の一定割合、方角に対しては一定の値を標準偏差としたガウス分布とする。
#      計測距離値が遠くなればなるほどばらつくような仕掛けとする。
# 　　 計測値が実際の値よりも、定常的に大きいまたは小さい値を出すようにする。    　
#
# ２）バイアス：常に計測値（距離・方角）に一定値を加えるもの
#   ⇒一様分布：計測値に対して、一定値だけずらす。系統誤差によるもの表現。
#　　　Sensorが生成されると決定される。距離に対しては一定の割合、方角に対しては一定値を加える
#
# ３）ファントム：計測レンジ内において見えないはずの物体を観測する
#   ⇒計測レンジ内の一様分布：偽のランドマークの値をドローして、極座標（r,θ）を返す
#     確率分布は、２項分布とし、一定の確率でファントムが発生する。また、ランドマークが多いほど、観測回数が増えるため
#　　　同時に、ファントムが観測される回数が増える仕組みとする。
#
# ４）見落とし：計測レンジ内において見えるはずの物体を見落とす
#   ⇒計測レンジ内の一様分布：ランドマーク極座標（r,θ）をNone返す
#
# ５）オクルージョン：ランドマークの一部が移動体により隠されてしまいセンサ値に影響が出る。
#   ⇒オクルージョンが発生するまでの時間期待値をλとし、
#    事象が発生する単位時間tにおける確率密度は、　P(X|λ)＝λexp(-λx)　で表される。
#    確率変数Xは、T=n*tのｎのことである。
#    また、オクルージョンは、計測した値よりも障害物で隠れてしまい、大きくなったり、小さくなったりしてしまうため、
#    計測値に計測レンジにはみ出さない値を（±）加える
#
class Camera(ICamera.IdealCamera):
    def __init__( self,env_map,time_interval,
    distance_range=(0.5,6.0),camera_phi_range=(-math.pi/3,math.pi/3),
    distance_noise_rate = 0.1,camera_phi_noise=math.pi/90,
    distance_bias_rate_stddev = 0.1,cameraphi_bias_stddev = math.pi/90,
    phantom_binomialprob = 0.01,phantom_Xrange =wo.World.getWorldRange(),phantom_Yrange =wo.World.getWorldRange(),
    oversight_binomialprob = 0.1,expected_occlusion_time=5 , occfarOcclusion_prob = 0.5):
        super().__init__(env_map,distance_range,camera_phi_range)
        
        self.observeds = []
        self.obsEvents = [] 

        self.distance_noise_rate = distance_noise_rate
        self.camera_phi_noise = camera_phi_noise

        self.dist_bias_rate_std = norm(scale=distance_bias_rate_stddev).rvs()
        self.cameraphi_bias_std = norm(scale=cameraphi_bias_stddev).rvs()
        
        rx,ry = phantom_Xrange,phantom_Yrange
        self.phantom_uniform_pdf = uniform(loc = (rx[0],ry[0]),scale=(rx[1]-rx[0],ry[1]-ry[0]))
        self.phantom_prob = phantom_binomialprob

        self.oversight_prob =oversight_binomialprob

        self.OccOcclusion_dst = expon(scale = expected_occlusion_time)
        self.Occlusion_time = self.OccOcclusion_dst.rvs()
        self.far_Occl_prob = occfarOcclusion_prob
        self.time_interval = time_interval

        
        

    def MeasuredValNoise(self,relpos):#計測値に対する雑音：ガウス性の雑音でセンサ値がばらつく
        #計測値を平均に、距離に対しては計測値の一定割合、方角に対しては一定の値を標準偏差としたガウス分布とする。
        addNoise_distance =  norm(loc = relpos[0],scale=relpos[0]*self.distance_noise_rate).rvs()
        addNoise_camphi   =  norm(loc = relpos[1],scale=self.camera_phi_noise).rvs()
        return np.array([addNoise_distance,addNoise_camphi]).T
    
    def addBisasNoise(self,relpos): # バイアス：常に計測値（距離・方角）に一定値を加えるもの
                                    #距離に対しては一定の割合、方角に対しては一定値を加える
        return relpos + np.array([ relpos[0] * self.dist_bias_rate_std , self.cameraphi_bias_std ]).T
    
    def ObservePhantom(self,relpos,cam_pose):#ファントム：計測レンジ内において見えないはずの物体を観測する
        if uniform.rvs() < self.phantom_prob:# 確率分布は、２項分布とし、一定の確率でファントムが発生する。
            self.obsEvents.append(KindofObsEventNoise.phantom)
            phantom_landark = np.array(self.phantom_uniform_pdf.rvs()).T#偽のランドマークの値をドローして、
            return self.observeation_function(cam_pose,phantom_landark)#極座標（r,θ）に変換する
        else:
            self.obsEvents.append(KindofObsEventNoise.noEvent)
            return relpos

    def Oversight(self,relpos):#見落とし：計測レンジ内において見えるはずの物体を見落とす
        if uniform.rvs() <  self.oversight_prob:# 確率分布は、２項分布とし、一定の確率の見逃しが発生する。
            self.obsEvents.append(KindofObsEventNoise.oversight)
            return None
        else:
            self.obsEvents.append(KindofObsEventNoise.noEvent)
            return relpos
    
    def OccOcclusion(self,relpos):#オクルージョン：ランドマークの一部が移動体により隠されてしまいセンサ値に影響が出る。    
        if self.Occlusion_time <=0 :#オクルージョンが発生するまでの時間
            self.Occlusion_time += self.OccOcclusion_dst.rvs()
            self.obsEvents.append(KindofObsEventNoise.occulusion)

            if uniform.rvs() < self.far_Occl_prob:#実際の計測値より遠くに見えた場合
                far_occ_r = relpos[0] + uniform.rvs()*(self.distance_range[1] - relpos[0])
                return np.array([far_occ_r,relpos[1]]).T
            else:                                 #実際の計測値より近くに見えた場合
                near_occ_r = relpos[0] + uniform.rvs()*( - relpos[0] )
                return np.array([near_occ_r,relpos[1]]).T
        
        else:
            self.obsEvents.append(KindofObsEventNoise.noEvent)
            return relpos


    def visible_bySensor(self,pairposes):#計測レンジ内であるか判定処理
        if pairposes is None: return False
        else:
            dis_frRtoLM  = pairposes[0]
            phi_frRtoLM  = pairposes[1]
          
            if( self.distance_range[0] <= dis_frRtoLM <= self.distance_range[1] and 
              self.camera_phi_range[0] <= phi_frRtoLM <= self.camera_phi_range[1] ):
                if KindofObsEventNoise.oversight in self.obsEvents:
                    return False

                else : 
                    return True

            else:
                self.obsEvents.append(KindofObsEventNoise.overRange)
                return False 

    def data(self,cam_pos):
        self.observeds = []
        self.Occlusion_time -= self.time_interval #オクルージョンが発生する時間の更新

        for lm in self.map.landmarks:
            self.obsEvents = []
            
            observed  = self.observeation_function(cam_pos,lm.pos)
            
            observed  = self.ObservePhantom(observed,cam_pos) #ファントムの写り込み
            observed  = self.OccOcclusion(observed)           #オクルージョンの写り込み
            self.Oversight(observed)                          #ランドマークの見落とし
            
            if self.visible_bySensor(observed) : #計測レンジ内もしくはOverSightであるか判定
                
                observed = self.MeasuredValNoise(observed) #計測値に生じるガウス性の雑音を加える
                observed = self.addBisasNoise(observed)#系統誤差であるバイアス（一定値）の雑音を加える

                self.observeds.append((observed,lm.id,self.obsEvents)) #タプルにして返す ([DIS,PHI],lANDMARK.id)　
                                                                       #理由：あえて変更不可にするため！
            else :
                self.observeds.append((None,None,self.obsEvents))

        self.obs_data = self.observeds

        return self.observeds


    def draw(self,ax,elems,cam_pos,robot_cirSize = 0) :#ax:描画するサブプロット実体 elems:グラフ図に描画するための図形リスト

        x_frRcom,y_frRcom,theta_frRcom = cam_pos #camera_poseだが、今回はWorld座標系のカメラの位置とロボットの位置は同一とみなしている
        
        for obs in self.obs_data:
            
            #観測イベントにより描画を変更する
            if obs[0] is None :
                obsEvent = obs[2]
                if KindofObsEventNoise.overRange not in obsEvent:
                    if KindofObsEventNoise.oversight in obsEvent:
                        elems.append(ax.text(-7.5,-7.5,"************\n"+"Occurred oversight\n************",fontsize=10))
            
            else:        
                dis_frRCom , phi_frRCom , obsEvent = obs[0][0],obs[0][1],obs[2]
           
                lx = x_frRcom + dis_frRCom * math.cos(theta_frRcom + phi_frRCom)
                ly = y_frRcom + dis_frRCom * math.sin(theta_frRcom + phi_frRCom)
                if  KindofObsEventNoise.phantom in obsEvent :
                    elems += ax.plot([x_frRcom,lx],[y_frRcom,ly],color="orange",linewidth = 3,linestyle="--") #MatPlotlibの特長で、plot([スタートX座標,X線分長さ],[スタートY座標,Y線分長さ])と書くと、
                                                                                                          #長さ範囲分だけ描画する
                    elems.append(ax.text(-7.5,-8.5,"************\n"+"Observed phantom\n************",fontsize=10))
                elif KindofObsEventNoise.occulusion in obsEvent:
                    elems += ax.plot([x_frRcom,lx],[y_frRcom,ly],color="blue",linewidth = 3,linestyle="--") #MatPlotlibの特長で、plot([スタートX座標,X線分長さ],[スタートY座標,Y線分長さ])と書くと、
                                                                                                          #長さ範囲分だけ描画する
                    elems.append(ax.text(-7.5,-6.5,"************\n"+"Occurred occulusion\n************",fontsize=10))    
                else :
                    elems += ax.plot([x_frRcom,lx],[y_frRcom,ly],color="green",linewidth = 3) #MatPlotlibの特長で、plot([スタートX座標,X線分長さ],[スタートY座標,Y線分長さ])と書くと、
                                                                                              #長さ範囲分だけ描画する
        
        #SensorRange範囲の描画
        lx_range1 = x_frRcom + (self.distance_range[1]-self.distance_range[0]) * math.cos(theta_frRcom + self.camera_phi_range[0])
        ly_range1 = y_frRcom + (self.distance_range[1]-self.distance_range[0]) * math.sin(theta_frRcom + self.camera_phi_range[0])
        lx_range2 = x_frRcom + (self.distance_range[1]-self.distance_range[0]) * math.cos(theta_frRcom + self.camera_phi_range[1])
        ly_range2 = y_frRcom + (self.distance_range[1]-self.distance_range[0]) * math.sin(theta_frRcom + self.camera_phi_range[1])
        
        lx_rangestart = x_frRcom + (robot_cirSize + self.distance_range[0]) * math.cos(theta_frRcom)
        ly_rangestart = y_frRcom + (robot_cirSize + self.distance_range[0]) * math.sin(theta_frRcom)
        
        elems += ax.plot([lx_rangestart,lx_range1],[ly_rangestart,ly_range1],color=(0, 0, 0.3, 0.4))
        elems += ax.plot([lx_rangestart,lx_range2],[ly_rangestart,ly_range2],color=(0, 0, 0.3, 0.4))
        x_rangefill = (lx_rangestart,lx_range1,lx_range2)
        y_rangefill = (ly_rangestart,ly_range1,ly_range2)
        elems += ax.fill(x_rangefill,y_rangefill,color = (0, 0, 0.2, 0.1)) #Sensor range内を塗りつぶす    
    
    


