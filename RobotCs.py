import IdealRobotCs as IRobot
import OccNoisemarkCs as Noisemark
from scipy.stats import expon , norm ,uniform
import math
import matplotlib.patches as patches
import numpy as np
from enum import Enum 
import WorldCs as wo

class KindofNoise(Enum):
    NoNoise ="-"
    Obstacle = "o"
    Bias = "b"
    Stuck = "s"
    kidnap = "k"

    #*************確率ロボティクスについて**************
    #
    # 状態方程式：Xt = f(Xt-1,μt) + εt
    # 
    # 確率的状態遷移モデル: Xt 〜 P(X|Xt-1,μt)
    # 
    # 
    # 　時刻tにおけるロボット座標は、”状態方程式”と”確率的状態遷移モデル”のどちらかで表現することができる。
    # ”状態方程式”は、「理想的な状態遷移があって、そこに雑音が発生し加わる」という発想から来ているものに対し、
    # ”確率的状態遷移モデル”は、「そもそもロボットの座標は不確定のものであり、一意に定まるものではない」という
    # 発想から来ている。
    # 　本質的には同じことを言っており、解釈の違いから来ているものであると考えら得れる。確率ロボティクスは、
    # 後者の、ロボット座標は不確定なものであるということを前提にすることで、学問体系的を構築している。
    # 
    #
class Robot(IRobot.IdealRobot):

    
    #実際のロボット走行におけるの状態遷移モデルは、下記の雑音が考えられる 
    #
    #１）雑音：突発的なロボットの向きの変化
    #          ex)小石など小物体を踏みつけたとき、走り出し、停止時のロボットの揺れなど
    #　　⇒指数分布
    # 
    #２）バイアス：制御指令値と実際のロボットへの制御出力値に対して常に一定の値がシフトしている
    #　　　　　　　ex)　継続：数秒〜数十秒：縁石への乗り上げ、走行環境の変化（斜面・滑る床）
    # 　　　　　　　　　継続：走行中ずっと：左右の車輪へかかる荷重バランス、モータ個体差、タイヤの状態など
    # 　 ⇒正規分布のランダム係数
    #
    #３）スタック：ロボットの同じ姿勢の停留
    #　　　　　　　雑音のレベルを超えたもの：走行ができなくなるような引っかかり
    # 　　　　　　　
    #４）誘拐：ロボットが別の場所へ突然ワープする（人が移動させたなど）
    #　　　　　雑音のレベルを超えたもの：人間の干渉
    #
    #

    def __init__(self,pose,agent=None,sensor=None,color="black"
    ,obstacles_per_meter = 10,steppedon_obst_std = math.pi/60,has_obsNoise=False
    ,randam_biasRatio=(0.1,0.1),expected_stuck_time=1e100,expected_escape_time=1e-100
    ,expected_kidnap_time=1e100, kidnap_Map_Xrange=wo.World.getWorldRange(),kidnap_Map_Yrange=wo.World.getWorldRange(),has_kidnap = False):
        super().__init__(pose,agent,sensor,color)
        
        self.LastOccNoisemarkInfo =  Noisemark.OccNoisemark()
        self.OccNoisemarkInfoList = []
        
        #*****走行中に小石やゴミを踏んでロボットの進行方向がズレる雑音モデル*****
        # 走行中に小さい障害物（小石など）を踏みつけて、ロボットの進行する向きがずれるという雑音モデルを考える
        #単位距離X（EX:１cmとか）あたりの踏みつける小さい障害物（小石など）の数の期待値(平均数)λとすると、
        #事象が発生する単位距離xにおける確率密度は、　P(X|λ)＝λexp(-λx)　で表される。
        #
        # また、同様に、踏みつける小さい障害物（小石など）の数の期待値(平均数)λを逆数1/λとすれば、
        #一つの小さい障害物（小石など）を踏みつけるまでの距離xにおける確率密度分布となる。
        #　P(X|1/λ)＝(1/λ)exp(-x/λ)
        # 　
        self.obsNoise_pdf = expon(scale = ( 1.0 / (1e-1000+steppedon_obst_std)) )  #一つ小さい障害物（小石など）を踏みつけるまでの距離xにおける確率密度  
        self.distance_untilSteppedonObsNoise = self.obsNoise_pdf.rvs()#一つ小さい障害物（小石など）を踏みつけるまでの確率変数である距離xをランダムに返す。
        self.thetabyObs_pdf = norm(steppedon_obst_std)#踏んだときに生じる実際の進行方向回転ズレは、ガウス分布に従うこととする。
        #　　　　　　　　　　　　　　　　　　　　　　　ガウス分布の回転ズレ確率密度分布となる
        self.Has_obsNoise = has_obsNoise #小さい障害物（小石など）を踏みつけた場合の雑音を加えるかのフラグ 
        # 
        #
        # *****バイアス：制御指令値（入力）と実際の出力を常に一定の大きさだけシフトするバイアス雑音モデル*****
        # 
        self.bias_randRatio_nu_pdf = norm(loc=1.0,scale= randam_biasRatio[0]) #速度指令値へ影響させるバイアス雑音のガウシアン確率密度分布
        self.bias_randRatio_omega_pdf = norm(loc=1.0,scale= randam_biasRatio[1])#回転指令値へ影響させるバイアス雑音のガウシアン確率密度分布
        #　　　　　　　　　　　　　　　　　　　　　　　ガウス分布の回転ズレ確率密度分布となる
        # 
        #
        # *****スタック：ロボットの同じ姿勢の停留 ：雑音のレベルを超えたもの：走行ができなくなるような引っかかり*****
        # 
        self.stuck_pdf  = expon(scale = expected_stuck_time) #ロボットにスタック（引っかかる）が生じるまでの時間期待値をλとし、
        #                                                        事象が発生する単位時間tにおける確率密度は、　P(X|λ)＝λexp(-λx)　で表される。
        self.escape_pdf = expon(scale = expected_escape_time)#スタック_ロボットが救助され復帰するまでの時間期待値をλとし、
        #                                                        事象が発生する単位時間tにおける確率密度は、　P(X|λ)＝λexp(-λx)　で表される。
        #
        self.time_until_stuckedRobot = self.stuck_pdf.rvs()#ロボットにスタック確率密度分布における確率変数である時刻tをランダムに返す。
        self.time_until_escapingRobotfromstuck = self.escape_pdf.rvs()#スタックしているロボットを救助される確率密度分布における確率変数である時刻tをランダムに返す。
        self.isStuck = False #ロボットがスタック中であるかのフラグ。
        #
        # *****誘拐：唐突な持ち去りによるロボット移動 ：雑音のレベルを超えたもの：人の介入*****
        #
        self.timekidnap_pdf = expon(scale = expected_kidnap_time)#ロボットが誘拐されるまでの時間期待値をλとし、
        #                                                        事象が発生する単位時間tにおける確率密度は、　P(X|λ)＝λexp(-λx)　で表される。
        self.time_until_kidnapRobot = self.timekidnap_pdf.rvs()#ロボットが誘拐される確率密度分布における確率変数である時刻tをランダムに返す。
        #                                                       ロボットが誘拐され運ばれる位置の確率密度分布は、Map内で一様分布にあたる。
        self.kidnap_dist = uniform(loc = (kidnap_Map_Xrange[0],kidnap_Map_Yrange[0],0),scale=(kidnap_Map_Xrange[1]-kidnap_Map_Xrange[0],kidnap_Map_Yrange[1]-kidnap_Map_Yrange[0],2*math.pi))
        self.Has_kidnap = has_kidnap


    #****誘拐：唐突な持ち去りによるロボット移動の雑音モデル
    def OccuredKidnap(self,pose,time_interval):
        self.time_until_kidnapRobot -=time_interval
        if  self.time_until_kidnapRobot <= 0:
            self.time_until_kidnapRobot +=self.timekidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose


    #*****バイアス：制御指令値（入力と実際の出力へのバイアス雑音モデル
    def Add_biasNoise(self,nu,omega):
        return nu * self.bias_randRatio_nu_pdf.rvs(), omega * self.bias_randRatio_omega_pdf.rvs()
    
    #*****スタック：ロボットの同じ姿勢の停留する雑音モデル
    def OccuredStuck(self,nu,omega,time_interval):
        if self.isStuck:
            self.time_until_escapingRobotfromstuck -= time_interval
            
            if self.time_until_escapingRobotfromstuck <= 0.0:
                self.time_until_escapingRobotfromstuck += self.escape_pdf.rvs()
                self.isStuck = False

        else:
            self.time_until_stuckedRobot -= time_interval
            if self.time_until_stuckedRobot <= 0.0 :
                self.time_until_stuckedRobot +=self.stuck_pdf.rvs()
                self.isStuck = True


        #スタック中は指令値(nu,omega)で進む量が0            
        if self.isStuck:
            return 0 , 0
        else:
            return nu  , omega

    
    #ロボットは進行・回転運動を行うので、時間間隔tに、指令速度μ*t と 角速度ωとロボット大きさr　を各々積し和とする。
    #時間間隔あたりのロボットが運動した距離を、小さい障害物（小石）を踏みつけるまでの残距離から引くことで、残距離を更新する    
    def OccuredNoise(self,pose,nu,omega,time_interval):
        self.distance_untilSteppedonObsNoise -= abs(nu) * time_interval + self.r * abs(omega) * time_interval 
        if(self.distance_untilSteppedonObsNoise <= 0.0):
                                                  #実際の回転ズレ雑音を発生させ、ロボットの姿勢に雑音を加える
            pose[2] +=  self.thetabyObs_pdf.rvs() #ロボットの姿勢に雑音を加える
            self.distance_untilSteppedonObsNoise += self.obsNoise_pdf.rvs() #雑音発生後、残距離をリセットするが、端数の影響を残すために
                                                                            # "+="　とする
        return pose
    
    
 
    def one_step(self,time_interval):
        if not self.agent:return
      
        tmp_nu = 0.0
        tmp_omega= 0.0
        obs = None

        if not self.sensor == None:#レンジファインダがある場合
            obs = self.sensor.data(self.pose)

        nu,omega = self.agent.decision(obs)
        
        Noise_code =""
        tmp_nu = nu
        tmp_omega = omega
        nu,omega = self.Add_biasNoise(nu,omega)#バイアス：制御指令値（入力と実際の出力へのバイアス雑音追加
        
        #if tmp_nu != nu or tmp_omega != omega:# バイアスは常にかかるので描画は不要
        #                                        :雑音発生時のアニメ描画のため雑音コード印字
        #    Noise_code +=KindofNoise.Bias.value
        #else:
        #    Noise_code +=KindofNoise.NoNoise.value

        
        tmp_nu = nu
        tmp_omega = omega
        nu,omega = self.OccuredStuck(nu,omega,time_interval)#スタック：ロボットの同じ姿勢の停留する雑音追加
        if tmp_nu != nu or tmp_omega != omega : #雑音発生時のアニメ描画のため雑音コード印字
            Noise_code +=KindofNoise.Stuck.value
        else:
            Noise_code +=KindofNoise.NoNoise.value


        self.pose = self.state_transition(nu,omega,time_interval,self.pose)#制御指令後のロボット姿勢変換
        
        if self.Has_obsNoise :#オブスタクル：走行中に小石やゴミを踏んでロボットの進行方向がズレる雑音追加*
            tmp_x,tmp_y,tmp_theta = self.pose[0],self.pose[1],self.pose[2]
            self.pose = self.OccuredNoise(self.pose,nu,omega,time_interval) 
            if tmp_theta != self.pose[2]  :#雑音発生時のアニメ描画のため雑音コード印字
                Noise_code +=KindofNoise.Obstacle.value
            else:
                Noise_code +=KindofNoise.NoNoise.value
        else:
                Noise_code +=KindofNoise.NoNoise.value
 
        if self.Has_kidnap :#誘拐：唐突な持ち去りによるロボット移動の雑音追加*
            tmp_x,tmp_y,tmp_theta = self.pose[0],self.pose[1],self.pose[2]
            self.pose = self.OccuredKidnap(self.pose,time_interval) 
            if tmp_x != self.pose[0] or tmp_y != self.pose[1] or tmp_theta != self.pose[2] :#雑音発生時のアニメ描画のため雑音コード印字
                Noise_code +=KindofNoise.kidnap.value
            else:
                Noise_code +=KindofNoise.NoNoise.value
        else:
                Noise_code +=KindofNoise.NoNoise.value

        if len(Noise_code) > 0 : #雑音情報の記録
            self.LastOccNoisemarkInfo.id = len(self.OccNoisemarkInfoList)
            self.LastOccNoisemarkInfo.pos = self.pose
            self.LastOccNoisemarkInfo.Noise_code= Noise_code
            self.LastOccNoisemarkInfo.Hasdrawn = False
            self.OccNoisemarkInfoList.append(self.LastOccNoisemarkInfo)
    

    def draw(self,ax,elems):
        x,y,theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn],[y,yn],color=self.color)
        c=patches.Circle(xy=(x,y),radius=self.r,fill=False,color=self.color)
        elems.append(ax.add_patch(c))
        
        #発生した雑音のマーキング
        self.LastOccNoisemarkInfo.draw(ax,elems)        

        self.poses.append(self.pose)
        
        elems += ax.plot( [e[0] for e in self.poses ],[e[1] for e in self.poses ],linewidth=0.5 ,color = "black" )
        if self.sensor and len(self.pose) > 1:
            self.sensor.draw(ax,elems,self.poses[-2],self.r) #Sensorオブジェクトには、ｔ-1時刻のロボット位置pose[t-1]から、制御指令Utしてからの
                                                            #ｔ時刻のロボット位置pose[t]となるので…   
                                                            # poses[0],poses[1] … poses[t-1],poses[t]
                                                            #よってリスト[-2]は、最後尾より二つ前なので、移動前のロボット位置pose[t-1]となる                                                 
        if hasattr(self.agent,"draw"):
                self.agent.draw(ax,elems)
        
        

        

        



