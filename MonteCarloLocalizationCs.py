from scipy.stats import expon , norm ,uniform ,multivariate_normal
import math
import matplotlib.patches as patches
import numpy as np
from enum import Enum 
import copy
import random

import ParticleCs as particle


class MonteCarloLocalization():
    def __init__(self,ini_pose,particles_num,motion_noise_stds={"nn":0.19,"no":0.001,"on":0.13,"oo":0.2}
    ,distance_dev_rate=0.14,direction_dev_rate=0.05,envmap = None):
        self.particles = [particle.Particle(ini_pose,1/particles_num) for i in range(particles_num)]
        
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_deve_rate = direction_dev_rate

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
        # 粒子を用いてのある範囲Xにおける確率密度P(x_t ∈　X)は
        # 
        # P(x^*_t ∈　X)　　=　Integral（(x∈X), bt(x)dx ) = 1/N (sum(δ（(x_t^(i) ∈　X)))
        # 
        # となる。δ（(x_t^(i) ∈　X)）はデルタ関数を表し、カッコ内の範囲(x_t^(i) ∈　X)に
        # いない場合は、「0」、いる場合は「1」となるものである。
        # 例えば、全粒子が30で、-5<=x<=5 の範囲に10個粒子がいた場合、
        # 　　　　　P(x^*_t ∈　X)　＝　10/30　＝　0.33　となる。
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
    

    def observeation_update(self,observeation): 
        for p in self.particles:#各粒子の観測の更新を行う
            p.observeation_update(observeation,self.map,self.distance_dev_rate,self.direction_deve_rate)
        
        #各粒子を重みの大きさに基づいてリサンプリング(Sinmple)する。
        #self.resampling_simple()
        
        #各粒子を重みの大きさに基づいてリサンプリング(系統サンプリング)する。
        self.resampling_systematic()

    def resampling_systematic(self):
        ws = np.cumsum( [p.weight for p in self.particles] ) #累積重みリストを作成する
        if ws[-1] < 1e-100 : ws = [ w + 1e-100 for w in ws]   #累積重みリストの最後の要素、つまり重みの和が0に丸め込まれると、
                                                            #ランダム選択が例外となるので、極小な値を加える
        
        step = ws[-1] / len(self.particles)#系統サンプリング用のステップサイズ（一様分布）を生成する
        r = np.random.uniform(0.0 , step)      #ステップサイズは、 (重みの総和) / 粒子数　とする。
        cur_pos = 0
        ps = []
            
        while ( len(ps) < len(self.particles) ):    #累積重みリストより、一様分布のステップ（初期値は０）分だけの要素
            if r < ws[cur_pos]:                     #を比較し、rより現在の重みリストの要素が小さかったら粒子を選択させる。
                ps.append(self.particles[cur_pos])  #選択後、rをStep分加えて、再び現在の要素に加える
                r += step                           #rより現在の重みリストの要素が大きかったら、要素番号を１すすめる。
            else:
                cur_pos +=1
        
        self.particles = [ copy.deepcopy(cp) for cp in ps]
        for p in self.particles : p.weight =  1.0 / len(self.particles)   #重さの大きさを更新する。
                                                                        #大きさは、粒子数の逆数となる。
                                                                        # ex: num=100, weight= 0.01
            


    def resampling_simple(self):
        ws = [p.weight for p in self.particles] #重みのリストを作成する
        if sum(ws) < 1e-100: #重みの和が0に丸め込まれると、ランダム選択が例外となるので、極小な値を加える
            ws = [ w + 1e-100 for w in ws]
        
        #現在の粒子の中で、重みの大きさに比例して、粒子を選ぶ（重複あり）
        #選択した粒子を用いて、再び更新する
        choicedParticles =random.choices(self.particles, weights = ws, k=len(self.particles))
        self.particles = [ copy.deepcopy(cp) for cp in choicedParticles]
        
        for p in self.particles: p.weight =  1.0 / len(self.particles)  #重さの大きさを更新する。
                                                                        #大きさは、粒子数の逆数となる。
                                                                        # ex: num=100, weight= 0.01
           


        



    def draw(self,ax,elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]

        #パーティクルの数が大きいいと平均の値も小さくなるので、補正するためにパーティクルの数もかけることにする。
        vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        elems.append(ax.quiver(xs,ys,vxs,vys,angles="xy",scale_units="xy",scale=1.5,color="navy",alpha = 0.4))
