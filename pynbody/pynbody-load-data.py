import pynbody as pb

fname = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.01000.gz'
#fname = '/home/ubuntu/projects/S2B/pynbody/run741CU/run741CU.00500.gz'
sim = pb.load(fname)
print(sim)