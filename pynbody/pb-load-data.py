import pynbody as pb

data_fname = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.00500.gz'
param_fname = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.param'

def pbload(filename, paramname=None):
    """
    Loads a snapshot using pynbody.  Can load a single species by appending
    ::gas, ::star, or ::dm to the filename
    
    Parameters
    ----------
    filename : str
        Filename to load
    paramname : str
        (optional) .param file to use
    
    Returns
    -------
    sim : snapshot
        A pynbody snapshot
    """
    if '::' in filename:
        filename, species = filename.split('::')
        sim = pb.load(filename, paramname=paramname)
        sim = getattr(sim, species)
    else:
        sim = pb.load(filename, paramname=paramname)
    return sim

if __name__ == '__main__':
    sim = pbload(data_fname,param_fname)
    print(sim)
