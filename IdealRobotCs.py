import math
import matplotlib.patches as patches
import numpy as np

class IdealRobot:
    def __init__(self,pose, agent = None,sensor = None, color="black"):
        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose] #時刻tごとにおけるロボット位置:poseのリスト　
                            #初期値は与えられた引数を予めアペンドする
        self.sensor = sensor

    
    
    def draw(self,ax,elems):
     x,y,theta = self.pose
     xn = x + self.r * math.cos(theta)
     yn = y + self.r * math.sin(theta)
     elems += ax.plot([x,xn],[y,yn],color=self.color)
     c=patches.Circle(xy=(x,y),radius=self.r,fill=False,color=self.color)
     elems.append(ax.add_patch(c))
     
     self.poses.append(self.pose)
     elems += ax.plot( [e[0] for e in self.poses ],[e[1] for e in self.poses ],linewidth=0.5 ,color = "black" )
     
     if self.sensor and len(self.pose) > 1:
       self.sensor.draw(ax,elems,self.poses[-2],self.r) #Sensorオブジェクトには、ｔ-1時刻のロボット位置pose[t-1]から、制御指令Utしてからの
                                                        #ｔ時刻のロボット位置pose[t]となるので…   
                                                        # poses[0],poses[1] … poses[t-1],poses[t]
                                                        #よってリスト[-2]は、最後尾より二つ前なので、移動前のロボット位置pose[t-1]となる                                                 

    #状態遷移関数 ロボット位置POSE(t-1)に制御指令Utである変化量を加えた後のロボットの位置POSE(t)を返す
    @classmethod
    def state_transition(cls,nu,omega,time,pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10: #角速度がほぼゼロのとき
          return pose + np.array([
                         nu*math.cos(t0),
                         nu*math.sin(t0),
                         omega
                        ]) *time
        else :
          return pose + np.array([
                        nu / omega *(math.sin(t0 + omega * time) - math.sin(t0)),
                        nu / omega *(-math.cos(t0 + omega * time) + math.cos(t0)),
                        omega * time
          ])


    def one_step(self,time_interval):
      if not self.agent:return
      
      obs = None
      if not self.sensor == None:#レンジファインダがある場合
       obs = self.sensor.data(self.pose)

      nu,omega = self.agent.decision(obs)
      self.pose = self.state_transition(nu,omega,time_interval,self.pose)
      
     



