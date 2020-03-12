import math
import numpy as np

class IdealCamera:
    def __init__(self,env_map, distance_range = (0.5,6.0),
        camera_phi_range = (-math.pi/3 , math.pi/3) ):
        
        self.map = env_map
        self.obs_data = []
        self.distance_range = distance_range
        self.camera_phi_range = camera_phi_range
    

    def visible_bySensor(self,pairposes):#計測レンジ内であるか判定処理
        if pairposes is None: return False
        else:
            dis_frRtoLM  = pairposes[0]
            phi_frRtoLM  = pairposes[1]
          
            if( self.distance_range[0] <= dis_frRtoLM <= self.distance_range[1] and 
              self.camera_phi_range[0] <= phi_frRtoLM <= self.camera_phi_range[1] ):
                return True
            else:
                return False 
                
            
    

    def data(self,cam_pos):
        observeds = []
        for lm in self.map.landmarks:
            observed  = self.observeation_function(cam_pos,lm.pos)
          
            if self.visible_bySensor(observed) : #計測レンジ内であるか判定
                observeds.append((observed,lm.id)) #タプルにして返す ([DIS,PHI],lANDMARK.id)　理由：あえて変更不可にするため！

        self.obs_data = observeds
        return observeds
    

    #観測方程式 : ロボット基準（カメラ基準：今回は同一としている）における距離と角度を返す
    @classmethod
    def observeation_function(cls,cam_pos,object_pos):
        diff = object_pos - cam_pos[0:2]        #pos[0] : x ,pos[1] : y ,pos[2] : θ
        phi = math.atan2(diff[1],diff[0]) - cam_pos[2]
        while phi >= np.pi : phi -=  2*np.pi
        while phi < -np.pi  : phi +=  2*np.pi  
        distance = math.sqrt(math.pow(diff[0],2)+math.pow(diff[1],2))
        return (np.array([distance,phi]).T).astype("float")

    def draw(self,ax,elems,cam_pos,robot_cirSize = 0) :#ax:描画するサブプロット実体 elems:グラフ図に描画するための図形リスト

        x_frRcom,y_frRcom,theta_frRcom = cam_pos #camera_poseだが、今回はWorld座標系のカメラの位置とロボットの位置は同一とみなしている
        
        for obs in self.obs_data:
           dis_frRCom , phi_frRCom = obs[0][0],obs[0][1]
           
           lx = x_frRcom + dis_frRCom * math.cos(theta_frRcom + phi_frRCom)
           ly = y_frRcom + dis_frRCom * math.sin(theta_frRcom + phi_frRCom)

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
    
    

