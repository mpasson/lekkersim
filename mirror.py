import numpy as np
import solver
import sys 

def connect(s1,p1,s2,p2):
    s1.add_conn(p1,s2,p2)    
    s2.add_conn(p2,s1,p1)    


def solve(st_list):
    while len(st_list)!=1:
        source_st=st_list[0].gone_to
        tar_st=st_list[0].connected_to[0].gone_to
        #print('Started join step')
        #print('Source structure: %50s inside %50s' % (source_st,source_st.gone_to))
        #print('Target structure: %50s inside %50s' % (tar_st,tar_st.gone_to))
        #print('\nSource connections')
        #source_st.print_conn()
        #print('\nTarget connections')
        #tar_st.print_conn()
        new_st=source_st.join(tar_st)
        st_list.remove(source_st)
        st_list.remove(tar_st)
        st_list.append(new_st)
        #print('\nGener. structure: %50s' % (new_st))
        #print('Generated connections')
        #new_st.print_conn()
        #print('\n')
    return st_list[0] 

r=0.5
C=10.0
for Lam in np.linspace(1.4,1.6,1001):
    

    BS1=solver.Structure(model=solver.GeneralBeamSplitter(Lam,ratio=r))
    BS2=solver.Structure(model=solver.GeneralBeamSplitter(Lam,ratio=r))
    BS3=solver.Structure(model=solver.GeneralBeamSplitter(Lam,ratio=0.5))

    WGD=solver.Structure(model=solver.waveguide(Lam,1.00,5.0))
    WGC11=solver.Structure(model=solver.waveguide(Lam,1.00,C))
    WGC12=solver.Structure(model=solver.waveguide(Lam,1.00,C))
    WGC21=solver.Structure(model=solver.waveguide(Lam,1.00,3*C))
    WGC22=solver.Structure(model=solver.waveguide(Lam,1.00,3*C))



    connect(BS1,'b0',WGD,'a0')  
    connect(BS2,'a0',WGD,'b0')  

    connect(BS1,'b1',WGC11,'a0')
    connect(BS3,'a0',WGC11,'b0')
    connect(BS2,'a1',WGC12,'a0')
    connect(BS3,'a1',WGC12,'b0')

    connect(BS3,'b0',WGC21,'a0')
    connect(BS1,'a1',WGC21,'b0')
    connect(BS3,'b1',WGC22,'a0')
    connect(BS2,'b1',WGC22,'b0')

    pin_mapping={
        'a0': (BS1,'a0'),
        'b0': (BS2,'b0'),
    }

    st_list=[BS1,BS2,BS3,WGD,WGC11,WGC12,WGC21,WGC22]
    full=solve(st_list)

    #print('\nObtained Structure')
    #full.print_pins()

    #print('\nNew structure from model')
    new=solver.Structure(model=full.get_model(pin_mapping))
    #new.print_pins()
    print('%15.8f %15.8f %15.8f' % (Lam,new.get_T('a0','a0'),new.get_T('a0','b0')))





