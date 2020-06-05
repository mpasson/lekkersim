import nazca as nd
import nazca.demofab as demo


with nd.Cell(name='testcell') as C:
    t=demo.shallow.strt(100.0).put(0,0,0)
    nd.Pin('a0').put(t.pin['a0'])
    t=demo.shallow.strt(100.0).put(t.pin['b0'])
    nd.Pin('b0').put(t.pin['b0'])

for params in nd.cell_iter(C):
    #print(params)
    if params.cell_close:
        continue
    for inode, [x, y, a], flip in params.iters['instance']:
        print(inode)        
        #print(dir(inode))
        print(inode.instance)
        print('')
