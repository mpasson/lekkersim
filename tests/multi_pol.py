import solver as sv
import numpy as np

np.set_printoptions(precision=4)

def func(wl,w,R,pol):
    return 2.0+0.1*pol


WG=sv.MultiPolWave(0.1,func,pol_list=[0,1],wl=1.55)
print(WG.pin_dic)
print(WG.create_S())
