import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from turb2d import RunMultiFlows
import time

# ipdb.set_trace()

proc = 10  # number of processors to be used
num_runs = 30
# Cmin, Cmax = [0.001, 0.03]
rmin, rmax = [10., 50.]
hmin, hmax = [10., 50.]
Cfmin, Cfmax = [0.024525, 0.3924]
mumin, mumax = [0.05, 0.3]

# C_ini = np.random.uniform(Cmin, Cmax, num_runs)
r_ini = np.random.uniform(rmin, rmax, num_runs)
h_ini = np.random.uniform(hmin, hmax, num_runs)
Cf_ini = np.random.uniform(Cfmin, Cfmax, num_runs)
mu_ini = np.random.uniform(mumin, mumax, num_runs)


rmf = RunMultiFlows(
    r_ini,
    h_ini,
    Cf_ini,
    mu_ini,
    'miroku_test220518_10.nc',
    processors=proc,
    endtime=500.0,
)
rmf.create_datafile()
start = time.time()
rmf.run_multiple_flows()
print("elapsed time: {} sec.".format(time.time() - start))
