from datetime import MAXYEAR
import pynbody as pb
import pynbody.plot.sph as sph

data_fname1 = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.00500.gz'
data_fname2 = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.01000.gz'
param_fname = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.param'
faceon_image = 'pynbody/images/run741CU-face-on1.png'
sideon_image = 'pynbody/images/run741CU-side-on1.png'
t_unit = 'Myr'
d_unit = 'kpc'
v_unit = 'km s^-1'
ro_unit = 'g cm^-3'
cmap = 'seismic'

def pbload(filename, paramname=None):
    if '::' in filename:
        filename, species = filename.split('::')
        sim = pb.load(filename, paramname=paramname)
        sim = getattr(sim, species)
    else:
        sim = pb.load(filename, paramname=paramname)
    return sim

def sim_explore(s):
    print(len(s.s),len(s.g),len(s.d))
    print()
    print(s.properties['time'].in_units(t_unit),t_unit)
    print()
    print(s.loadable_keys())
    print()
    print(s.s['pos'].in_units(d_unit))
    print()
    print(s.s['vel'].in_units(v_unit))
    print(s.physical_units())
    print()

def sim_images(s):
    pb.analysis.angmom.faceon(s)
    s.rotate_z(90)
    sph.image(s.star,qty="rho",units=ro_unit,width=50,cmap=cmap,filename=faceon_image)
    s.rotate_x(90)
    sph.image(s.star,qty="rho",units=ro_unit,width=50,cmap=cmap,filename=sideon_image)

    #pb.analysis.angmom.sideon(s)
    #sph.image(s.star,qty="rho",units="g cm^-3",width=50,cmap="seismic",filename=sideon_image)


if __name__ == '__main__':
    s = pbload(data_fname1,param_fname)
    
    sim_explore(s)
    x = s.s['pos'][:,0].in_units(d_unit)
    y = s.s['pos'][:,1].in_units(d_unit)
    z = s.s['pos'][:,2].in_units(d_unit)
    vx = s.s['vel'][:,0].in_units(v_unit)
    vy = s.s['vel'][:,1].in_units(v_unit)
    vz = s.s['vel'][:,2].in_units(v_unit)


