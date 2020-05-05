import numpy as np
import solver
import sys 


r=0.5

#BS1=solver.Structure(model=solver.Splitter1x2())
#BS2=solver.Structure(model=solver.Splitter1x2())

BS1=solver.Structure(model=solver.BeamSplitter(phase=0.5))
BS2=solver.Structure(model=solver.BeamSplitter(phase=0.5))

WG1=solver.Structure(model=solver.waveguide(20.0))
WG2=solver.Structure(model=solver.waveguide(20.0))

SP1=solver.Structure(model=solver.Splitter1x2())
SP2=solver.Structure(model=solver.Splitter1x2())

#Sol=solver.Solver(structures=[BS1,BS2,WG1,WG2])
Sol=solver.Solver(structures=[SP1,BS2,WG1,WG2])
#Sol=solver.Solver(structures=[SP1,SP2,WG1,WG2])


#Sol.connect(BS1,'b0',WG1,'a0')  
#Sol.connect(BS1,'b1',WG2,'a0')
Sol.connect(SP1,'b0',WG1,'a0')  
Sol.connect(SP1,'b1',WG2,'a0')

Sol.connect(BS2,'b0',WG1,'b0')
Sol.connect(BS2,'b1',WG2,'b0')
#Sol.connect(SP2,'b0',WG1,'b0')
#Sol.connect(SP2,'b1',WG2,'b0')

'''
for p in np.linspace(0.0,1.0,101):
    input_dic={'a0':np.exp(1.0j*np.pi*p)}
    out1=BS1.get_output(input_dic,power=False)
    input_dic={'a1':np.exp(1.0j*np.pi*p)}
    out2=BS1.get_output(input_dic,power=False)

    print(5*'%8.4f' % (p, np.angle(out1['b0'])/np.pi, np.angle(out1['b1'])/np.pi, np.angle(out2['b0'])/np.pi, np.angle(out2['b1'])/np.pi))
quit()
'''
for i in [0.0,1.0]:
    for j in [0.0,1.0]:
        input_dic={'a0':i, 'a1':j}
        out1=BS1.get_output(input_dic,power=False)
        print(6*'%8.4f' % (i,j, out1['b0'].real, out1['b0'].imag, out1['b1'].real, out1['b1'].imag))

print('')
quit()
for i in [0.0,1.0]:
    for j in [0.0,1.0]:
        input_dic={'b0':i+1.0j*j}
        out1=SP1.get_output(input_dic,power=False)
        print(4*'%8.4f' % (i,j, out1['a0'].real, out1['a0'].imag))


quit()





pin_mapping={
#    'a0': (BS1,'a0'),
#    'a1': (BS1,'a1'),
    'a0': (SP1,'a0'),
    'b0': (BS2,'a0'),
    'b1': (BS2,'a1'),
#    'b0': (SP2,'a0'),
}

Sol.map_pins(pin_mapping)


    
Sol.set_param('Lam',value=1.55)
full=Sol.solve()
new=full.get_model(pin_mapping)
#print(5*'%15.8f' % (1.55,new.get_T('a0','b0'),new.get_T('a1','b1'),new.get_T('a0','b1'),new.get_T('a1','b0')))
#print(5*'%15.8f' % (1.55,new.get_T('a0','b0'),new.get_T('a0','b1'),new.get_T('b0','a0'),new.get_T('b1','a0')))
#print(2*'%15.8f' % (1.55,new.get_T('a0','b0')))

for p in np.linspace(0.0,1.0,101):
    input_dic={'b0':1.0,'b1':np.exp(1.0j*np.pi*p)}
    out=new.get_output(input_dic)
    print('%8.4f %8.4f' % (p,out['a0']))
