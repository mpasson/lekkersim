import solver
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize, precision=3, linewidth=200)

FPR=solver.FPR_NxM(2,2)

def AWG(Nin,Nmid,Nout,L0,DL,order,neff):
    with solver.Solver() as AWG:
        FPR1=solver.FPR_NxM(Nin,Nmid,phi=0.2).put()
        FPR2=solver.FPR_NxM(Nmid,Nout,phi=0.2).put()

        for i in range(Nin):
            solver.putpin(f'a{i}',FPR1.pin[f'a{i}'])       
        for i in range(Nout):
            solver.putpin(f'b{i}',FPR2.pin[f'b{i}'])
        for i in range(Nmid):
            wg=solver.Waveguide(L=L0+order*(i-0.5*Nmid+0.5)*DL,n=neff).put('a0',FPR1.pin[f'b{i}'])
            solver.connect(wg.pin['b0'],FPR2.pin[f'a{i}'])

    return AWG

Ni=9
No=9

op=[]
for i in range(Ni):
    op.append(open(f'AWG_a{i}.out','w'))

AW=AWG(Ni,21,No,501.0,0.516,50,2.3)     

for Lam in np.linspace(1.27,1.32,1001):
    
    AW.set_param('Lam',value=Lam)  
    M=AW.solve()
    for i in range(Ni):
            st=f'{Lam:12.6f}'
            for j in range(No):
                T=M.get_T(f'a{i}',f'b{j}')
                st=st+f' {T:12.6f}'
            st=st+'\n'
            op[i].write(st)          

    #print(6*'%12.6f' % (Lam,M.get_T('a2','b0'),M.get_T('a2','b1'),M.get_T('a2','b2'),M.get_T('a2','b3'),M.get_T('a2','b4')))      

for f in op:
    f.close()
