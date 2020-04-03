import solver


def connect(s1,p1,s2,p2):
    s1.add_conn(p1,s2,p2)    
    s2.add_conn(p2,s1,p1)    


def solve(st_list):
    while len(st_list)!=1:
        print(st_list)
        source_st=st_list[0].gone_to
        tar_st=st_list[0].connected_to[0].gone_to
        new_st=source_st.join(tar_st)
        st_list.remove(source_st)
        st_list.remove(tar_st)
        st_list.append(new_st)
    return st_list[0]    


WG1=solver.Structure(model=solver.waveguide(1.55,1.45,10.0))
  
BS1=solver.Structure(model=solver.BeamSplitter(1.55))

connect(BS1,'b0',WG1,'a0')

print('\nWaveguide:')
print(WG1.pin_dic)
print(WG1.Smatrix)

print('\nBeam splitter:')
print(BS1.pin_dic)
print(BS1.Smatrix)

MERGED=BS1.join(WG1)

quit()

connect(BS1,'b0',WG1,'a0')  
connect(BS1,'b1',WG2,'a0')
connect(BS2,'a0',WG1,'b0')
connect(BS2,'a1',WG2,'b0')

st_list=[BS1,BS2,WG1,WG2]
solve(st_list)


