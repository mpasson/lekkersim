import numpy as np
import pytest as pt

import matplotlib.pyplot as plt

import nazca as nd
from nazca import demofab as demo
import solver as sv

@pt.fixture
def no_pol():
    MMI=demo.mmi2x2_sh()
    demo.xsShallow.pol_list=None
    return demo.xsShallow,MMI

@pt.fixture
def two_pol():
    def func(wl=None,R=None,w=None,pol=None):
        return 2.5+0.5*pol
    demo.xsShallow.set_Neff(func)
    demo.xsShallow.pol_list=[0,1]
    MMI=demo.mmi2x2_sh()
    MMI.model_info['model']=sv.GeneralBeamSplitter(ratio=0.5,phase=0.5).expand_pol([0,1])
    return demo.xsShallow,MMI



def add_drop(radius):
    with nd.Cell(name=f'Shallow_AddDrop_R{radius:.4}') as C:
        M1=MMI.put()
        demo.shallow.bend(angle=180.0,radius=radius).put()
        M2=MMI.put()    
        demo.shallow.bend(angle=180.0,radius=radius).put()
        nd.Pin('a0',pin=M1.pin['a1']).put()
        nd.Pin('a1',pin=M2.pin['b1']).put()
        nd.Pin('b0',pin=M1.pin['b1']).put()
        nd.Pin('b1',pin=M2.pin['a1']).put()
    return C



def test_single(no_pol):
    strt=demo.shallow.strt(100.0)
    sol=nd.get_solver(strt)
    mod=sol.solve(wl=1.55)
    assert mod.get_T('a0','b0') == pt.approx(1.0, 1e-8)

def test_1levl_circuit(no_pol):
    xsShallow,MMI=no_pol

    def add_drop(radius):
        with nd.Cell(name=f'Shallow_AddDrop_R{radius:.4}') as C:
            M1=MMI.put()
            demo.shallow.bend(angle=180.0,radius=radius).put()
            M2=MMI.put()    
            demo.shallow.bend(angle=180.0,radius=radius).put()
            nd.Pin('a0',pin=M1.pin['a1']).put()
            nd.Pin('a1',pin=M2.pin['b1']).put()
            nd.Pin('b0',pin=M1.pin['b1']).put()
            nd.Pin('b1',pin=M2.pin['a1']).put()
        return C

    add_drop_1=add_drop(20.0)
    sol=nd.get_solver(add_drop_1)

    wl_l=np.linspace(1.54,1.56,201)
    T=[ sol.solve(wl = wl).get_T('a0', 'b0') for wl in wl_l ]
    t=np.sqrt(0.5)
    a=-np.sqrt(0.5)
    ex=np.array([np.exp(-4.0j*np.pi**2.0/wl*demo.xsShallow.Neff(wl=wl,R=20.0)*20.0) for wl in wl_l ])
    Tref=np.abs((-a+t*ex)/(-a*t+ex))**2.0
    assert np.allclose(T,Tref)

def test_2levl_circuit(no_pol):
    xsShallow,MMI=no_pol

    def add_drop(radius):
        with nd.Cell(name=f'Shallow_AddDrop_R{radius:.4}') as C:
            M1=MMI.put()
            demo.shallow.bend(angle=180.0,radius=radius).put()
            M2=MMI.put()    
            demo.shallow.bend(angle=180.0,radius=radius).put()
            nd.Pin('a0',pin=M1.pin['a1']).put()
            nd.Pin('a1',pin=M2.pin['b1']).put()
            nd.Pin('b0',pin=M1.pin['b1']).put()
            nd.Pin('b1',pin=M2.pin['a1']).put()
        return C

    ad_R10=add_drop(20.0)
    ad_R12=add_drop(25.0)

    with nd.Cell(name='Double_Filter') as DF:
        r1=ad_R10.put()
        demo.shallow.strt(300.0).put(r1.pin['a1'])
        r2=ad_R12.put(flip=True)
        
        nd.Pin('a0',pin=r1.pin['a0']).put()
        nd.Pin('a1',pin=r2.pin['b0']).put()
        nd.Pin('a2',pin=r2.pin['b1']).put()
        nd.Pin('b0',pin=r1.pin['b0']).put()
        nd.Pin('b1',pin=r1.pin['b1']).put()
        nd.Pin('b2',pin=r2.pin['a1']).put()

    sol=nd.get_solver(DF, fullreturn=True)

    wl_l=np.linspace(1.54,1.56,201)
    t=np.sqrt(0.5)
    a=-np.sqrt(0.5)

    ex=np.array([np.exp(-4.0j*np.pi**2.0/wl*demo.xsShallow.Neff(wl=wl,R=20.0)*20.0) for wl in wl_l ])
    Tref1=np.abs((-a+t*ex)/(-a*t+ex))**2.0
    ex=np.array([np.exp(-4.0j*np.pi**2.0/wl*demo.xsShallow.Neff(wl=wl,R=25.0)*25.0) for wl in wl_l ])
    Tref2=np.abs((-a+t*ex)/(-a*t+ex))**2.0

    T1=[ sol[ad_R10.cnode].solve(wl = wl).get_T('a0', 'a1') for wl in wl_l ]
    T2=[ sol[ad_R12.cnode].solve(wl = wl).get_T('a0', 'a1') for wl in wl_l ]
    T3=[ sol[DF.cnode].solve(wl = wl).get_T('a0', 'b2') for wl in wl_l ]

    assert np.allclose(T1,1.0-Tref1)
    assert np.allclose(T2,1.0-Tref2)
    assert np.allclose(T3,(1.0-Tref1)*(1.0-Tref2))


def test_params(no_pol):
    xsShallow,MMI=no_pol

    with nd.Cell(name='PhaseShifter') as PS:
        wg=demo.shallow.strt(100.0).put(5.0,0.0,0.0)
        nd.Pin('a0',pin=wg.pin['a0']).put()
        nd.Pin('b0',pin=wg.pin['b0']).put()
        PS.model_info['model']=sv.TH_PhaseShifter(100.0,xsShallow.Neff)

    with nd.Cell(name='MZM') as MZM_bal:
        m1=MMI.put()
        demo.shallow.sbend(offset=50.0).put()
        ps=PS.put(param_mapping={'PS': 'PS1'})
        demo.shallow.sbend(offset=-50.0).put()
        m2=MMI.put()
        demo.shallow.sbend(offset=-50.0).put(m1.pin['b1'])
        ps2=PS.put(param_mapping={'PS': 'PS2'})
        demo.shallow.sbend(offset=50.0).put()
        
        nd.Pin('a0',pin=m1.pin['a0']).put()
        nd.Pin('a1',pin=m1.pin['a1']).put()
        nd.Pin('b0',pin=m2.pin['b0']).put()
        nd.Pin('b1',pin=m2.pin['b1']).put()


    sol=nd.get_solver(MZM_bal)

    psl=np.linspace(0.0,1.0,101)
    T1=[sol.solve(wl=1.55,PS1=ps).get_T('a0','b0') for ps in psl]
    T2=[sol.solve(wl=1.55,PS1=ps,PS2=0.5).get_T('a0','b0') for ps in psl]
    assert np.allclose(T1,np.sin(0.5*np.pi*psl)**2.0)
    assert np.allclose(T2,np.sin((0.5*psl-0.25)*np.pi)**2.0)


def test_other_cells(no_pol):
    xsShallow,MMI=no_pol
    with nd.Cell(name='PhaseShifter_wp') as PS:
        t1=demo.metaldc.taper(width2=2.0,length=10.0).put()
        demo.metaldc.strt(90.0,width=2.0).put()
        t2=demo.metaldc.taper(width1=2.0,length=10.0).put()
        
        wg=demo.shallow.strt(100.0).put(5.0,0.0,0.0)
        
        nd.Pin('c0',pin=t1.pin['a0']).put()
        nd.Pin('c1',pin=t2.pin['b0']).put()
        nd.Pin('a0',pin=wg.pin['a0']).put()
        nd.Pin('b0',pin=wg.pin['b0']).put()
        
        PS.model_info['model']=sv.TH_PhaseShifter(100.0,xsShallow.Neff)
        
    with nd.Cell(name='DC_pad') as DCp:
        demo.metaldc.strt(100.0,width=100.0).put(-50.0,0,0)
        nd.Pin('a0',xs=demo.metaldc.xs).put(0,0,90.0)

    with nd.Cell(name='MZM_wp') as MZM_bal:
        m1=MMI.put()
        demo.shallow.sbend(offset=50.0).put()
        ps=PS.put(param_mapping={'PS': 'PS1'})
        demo.shallow.sbend(offset=-50.0).put()
        m2=MMI.put()
        demo.shallow.sbend(offset=-50.0).put(m1.pin['b1'])
        ps2=PS.put(param_mapping={'PS': 'PS2'})
        demo.shallow.sbend(offset=50.0).put()
        
        DCp.put(400.0,300.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps.pin['c0']).put()
        DCp.put(800.0,300.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps.pin['c1']).put()

        DCp.put(200.0,300.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps2.pin['c0']).put()
        DCp.put(1000.0,300.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps2.pin['c1']).put()
        
        
        nd.Pin('a0',pin=m1.pin['a0']).put()
        nd.Pin('a1',pin=m1.pin['a1']).put()
        nd.Pin('b0',pin=m2.pin['b0']).put()
        nd.Pin('b1',pin=m2.pin['b1']).put()

    sol=nd.get_solver(MZM_bal)

    psl=np.linspace(0.0,1.0,101)
    T1=[sol.solve(wl=1.55,PS1=ps).get_T('a0','b0') for ps in psl]
    T2=[sol.solve(wl=1.55,PS1=ps,PS2=0.5).get_T('a0','b0') for ps in psl]
    assert np.allclose(T1,np.sin(0.5*np.pi*psl)**2.0)
    assert np.allclose(T2,np.sin((0.5*psl-0.25)*np.pi)**2.0)


def test_twopol_basic(two_pol):
    xsShallow,MMI=two_pol

    strt=demo.shallow.strt(100.0)
    sol=nd.get_solver(strt)
    mod=sol.solve(wl=1.55)
    assert mod.get_T('a0_pol0','b0_pol0') == pt.approx(1.0, 1e-8)
    assert mod.get_T('a0_pol1','b0_pol1') == pt.approx(1.0, 1e-8)


def test_twopol_MZM(two_pol):
    xsShallow,MMI=two_pol

    with nd.Cell(name='PhaseShifter') as PS:
        t1=demo.metaldc.taper(width2=2.0,length=10.0).put()
        demo.metaldc.strt(90.0,width=2.0).put()
        t2=demo.metaldc.taper(width1=2.0,length=10.0).put()
        
        wg=demo.shallow.strt(100.0).put(5.0,0.0,0.0)
        
        nd.Pin('c0',pin=t1.pin['a0']).put()
        nd.Pin('c1',pin=t2.pin['b0']).put()
        nd.Pin('a0',pin=wg.pin['a0']).put()
        nd.Pin('b0',pin=wg.pin['b0']).put()
        
        with sv.Solver(name='Multipol_ps') as SOL:
            WGm=sv.MultiPolWave(90.0,xsShallow.Neff,pol_list=[0,1])
            PSm=sv.PhaseShifter().expand_pol(pol_list=[0,1])
            wg=WGm.put()
            ps=PSm.put()

            sv.connect(wg.pin['b0_pol0'],ps.pin['a0_pol0'])
            sv.connect(wg.pin['b0_pol1'],ps.pin['a0_pol1'])
            
            sv.Pin('a0_pol0').put(wg.pin['a0_pol0'])
            sv.Pin('a0_pol1').put(wg.pin['a0_pol1'])
            sv.Pin('b0_pol0').put(ps.pin['b0_pol0'])
            sv.Pin('b0_pol1').put(ps.pin['b0_pol1'])


        PS.model_info['model']=SOL
        
    with nd.Cell(name='DC_pad') as DCp:
        demo.metaldc.strt(100.0,width=100.0).put(-50.0,0,0)
        nd.Pin('a0',xs=demo.metaldc.xs).put(0,0,90.0)

    with nd.Cell(name='MZM') as MZM_bal:
        m1=MMI.put()
        demo.shallow.sbend(offset=410.0).put()
        ps=PS.put(param_mapping={'PS': 'PS1'})
        demo.shallow.sbend(offset=-410.0).put()
        m2=MMI.put()
        demo.shallow.sbend(offset=-400.0).put(m1.pin['b1'])
        ps2=PS.put(param_mapping={'PS': 'PS2'})
        demo.shallow.sbend(offset=400.0).put()
        
        DCp.put(600.0,600.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps.pin['c0']).put()
        DCp.put(1000.0,600.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps.pin['c1']).put()

        DCp.put(400.0,600.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps2.pin['c0']).put()
        DCp.put(1200.0,600.0,180.0)
        demo.metaldc.strt_bend_strt_p2p(ps2.pin['c1']).put()
        
        
        nd.Pin('a0',pin=m1.pin['a0']).put()
        nd.Pin('a1',pin=m1.pin['a1']).put()
        nd.Pin('b0',pin=m2.pin['b0']).put()
        nd.Pin('b1',pin=m2.pin['b1']).put()

    sol=nd.get_solver(MZM_bal, infolevel=0, drc=True)


    wll=np.linspace(1.5,1.6,101)
    T0_1=[sol.solve(wl=wl,PS1=0.0,PS2=0.0).get_T('a0_pol0','b0_pol0') for wl in wll]
    T0_2=[sol.solve(wl=wl,PS1=0.0,PS2=0.5).get_T('a0_pol0','b0_pol0') for wl in wll]
    T1_1=[sol.solve(wl=wl,PS1=0.0,PS2=0.0).get_T('a0_pol1','b0_pol1') for wl in wll]
    T1_2=[sol.solve(wl=wl,PS1=0.0,PS2=0.5).get_T('a0_pol1','b0_pol1') for wl in wll]

    assert np.allclose(T0_1,np.sin((2.5/wll*20.0)*np.pi)**2.0)
    assert np.allclose(T0_2,np.sin((2.5/wll*20.0-0.25)*np.pi)**2.0)
    assert np.allclose(T1_1,np.sin((3.0/wll*20.0)*np.pi)**2.0)
    assert np.allclose(T1_2,np.sin((3.0/wll*20.0-0.25)*np.pi)**2.0)


if __name__=='__main__':
    pass
