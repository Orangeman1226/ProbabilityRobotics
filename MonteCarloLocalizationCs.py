from scipy.stats import expon , norm ,uniform ,multivariate_normal
import math
import matplotlib.patches as patches
import numpy as np
from enum import Enum 

import ParticleCs as particle


class MonteCarloLocalization():
    def __init__(self,ini_pose,particles_num,motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2}):
        self.particles = [particle.Particle(ini_pose) for i in range(particles_num)]
        
        #
        # 　ロボットの挙動をエージェントが外から毎時刻統計でしらべて、状態を遷移する状態遷移モデルを考える。
        # ロボットの動きに生じる雑音の大きさは、ガウス分布に従うと仮定し、その分散は移動量（道のりや向き変化量）
        # に比例すると仮定する。生成した粒子には毎ターン雑音を混ぜることとする。
        # 
        # 　雑音σ_abは、4つのパラメータによりなることとする。
        # 　１）σ_nn :直進１mで生じる道のりのばらつきの標準偏差
        #   ２）σ_no :回転１radで生じる道のりのばらつきの標準偏差
        #   ３）σ_on :直進１mで生じる向きのばらつきの標準偏差
        #   ４）σ_oo :回転１radで生じる向きのばらつきの標準偏差
        #
        #   １）ー４）の雑音をまとめ一般化するとσ_abと表され、「ｂで生じるａの標準偏差」ということになる。 　
        #
        # 　　ここで、σ_ab　〜　N（0,σ_ab^2）のガウス分布従うので…
        # 　移動量n[m](もしくはo[rad])が生じると、雑音σ_abの分散も比例する（分散の性質）。
        # 　よって、比の関係より
        # 　　　　　σ_ab^2　：　(σ'_ab・⊿t)^2  ＝　１　：　|b|⊿t
        #                   
        #               σ'_ab = σ_ab・squrt ( ⊿t / |b|  )
        # 　となる。
        #
        #   以上より、制御指令値U=(nu,omega)とすると、粒子の移動U'=(n',o') をもとめると・・・
        #             n=nu,o=omegaとする
        # 　　　
        #      [ n' ] = [n] + [ σ_nn・squrt ( ⊿t / |n|  ) + σ_nn・squrt ( ⊿t / |n|  ) ] 
        #  　  [ o' ] = [o] + [ σ_on・squrt ( ⊿t / |b|  ) + σ_oo・squrt ( ⊿t / |o|  ) ]
        # 
        # 
        self.mNoise_stds = motion_noise_stds #粒子が動くときに生じるノイズ：4次元ガウス分布となる。
                                             #共分散行列用の標準偏差と共分散行列
        self.mNoise_covariance = np.diag([self.mNoise_stds["nn"]**2,self.mNoise_stds["no"]**2
                                         ,self.mNoise_stds["on"]**2,self.mNoise_stds["oo"]**2])
        self.motionNoise_rate_pdf = multivariate_normal(cov=self.mNoise_covariance)#四次元ガウス分布
    

    def motion_update(self,nu,omega,time):
        for p in self.particles:#各粒子の移動の更新を行う
            p.motion_update(nu,omega,time,self.motionNoise_rate_pdf)


    def draw(self,ax,elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]

        vxs = [math.sin(p.pose[2]) for p in self.particles]
        vys = [math.cos(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs,ys,vxs,vys,color="navy",alpha = 0.4))
