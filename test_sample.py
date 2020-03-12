import math
import numpy as np
import EstimationAgentCS as EstAgent
import MonteCarloLocalizationCs as MCL

start_pos=np.array([0,0,0]).T
particlsNum = 100
estimator = MCL.MonteCarloLocalization(start_pos,particlsNum,motion_noise_stds={"nn":0.01,"no":0.02,"on":0.03,"oo":0.04})
a = EstAgent.EstimationAgent( 0.1,0.2,10.0/180 * math.pi,estimator)
estimator.motion_update(0.2,10.0/180 * math.pi,0.1)

x = []
y = []
t = []
for p in estimator.particles:
    print(p.pose)
    x.append(p.pose[0])
    y.append(p.pose[1])
    t.append(p.pose[2])

print("x_ave="+ str(np.average(x)))
print("y_ave="+ str(np.average(y)))
print("t_ave="+ str(np.average(t)))

    