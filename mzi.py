import numpy as np
import solver

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

for Lam in np.linspace(1.4,1.6,201):
    

    BS1=solver.Structure(model=solver.BeamSplitter(Lam))
    BS2=solver.Structure(model=solver.BeamSplitter(Lam))

    WG1=solver.Structure(model=solver.waveguide(Lam,1.00,20.0))
    WG2=solver.Structure(model=solver.waveguide(Lam,1.00,10.0))

    connect(BS1,'b0',WG1,'a0')  
    connect(BS1,'b1',WG2,'a0')
    connect(BS2,'a0',WG1,'b0')
    connect(BS2,'a1',WG2,'b0')

    pin_mapping={
        'a0': (BS1,'a0'),
        'a1': (BS1,'a1'),
        'b0': (BS2,'b0'),
        'b1': (BS2,'b1'),
    }

    st_list=[BS1,BS2,WG1,WG2]
    full=solve(st_list)

    #print('\nObtained Structure')
    #full.print_pins()

    #print('\nNew structure from model')
    new=solver.Structure(model=full.get_model(pin_mapping))
    #new.print_pins()
    print('%15.8f %15.8f' % (Lam,new.get_T('a0','b0')))





