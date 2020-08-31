import solver as sv
import numpy as np

with sv.Solver() as S:
    WG=sv.Waveguide(0.05,2.5)
    for i in range(10):
        wg=WG.put()
        sv.Pin(f'a{i}').put(wg.pin['a0'])
        for j in range(i):
            wg=WG.put('a0',wg.pin['b0'])
        sv.Pin(f'b{i}').put(wg.pin['b0'])

with sv.Solver() as S2:
    WG=sv.Waveguide(10.0,2.5)

    wg1=WG.put()
    wg2=WG.put()

    wg1.join(wg2)



M=S.solve(wl=1.55)
pri=''
for i in range(10):
    for j in range(10):
        T=M.get_A(f'a{i}',f'b{j}')
        pri+=f' ({T.real:5.3f},{T.imag:5.3f}) '
    pri+='\n'
print(pri)
