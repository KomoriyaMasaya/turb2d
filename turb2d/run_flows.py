import os
from matplotlib.pyplot import axis
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from turb2d import TurbidityCurrent2D
from turb2d.utils import create_topography, create_init_flow_region, create_topography_from_geotiff
import numpy as np

import time
import multiprocessing as mp
import netCDF4
from landlab.io.native_landlab import save_grid
from tqdm import tqdm 

class RunMultiFlows():
    """A class to run multiple flows for conducting inverse analysis
    """
    def __init__(
            self,
            r_ini,
            h_ini,
            Cf_ini,
            mu_ini,
            filename,
            processors=1,
            endtime=1000,
    ):

        self.r_ini = r_ini
        self.h_ini = h_ini
        self.Cf_ini = Cf_ini
        self.mu_ini = mu_ini
        self.filename = filename
        self.processors = processors
        self.endtime = endtime
        self.num_runs = r_ini.shape[0]

    def produce_flow(self, r_ini, h_ini, Cf_ini, mu_ini):
        """ producing a TurbidityCurrent2D object.
        """

        # create a grid

        grid = create_topography_from_geotiff('merge10x10.tif',
                                    xlim=[640,880],
                                    ylim=[260,700],
                                    spacing=5,
                                    filter_size=[5, 5])
        #tpography is reversed
        #tc.eta = np.fliplr(tc.eta)
        #grid = np.flip(grid, axis=1)
        #import pdb
        #pdb.set_trace()

        # grid = create_topography(
        #     length=5000,
        #     width=2000,
        #     spacing=10,
        #     slope_outside=0.2,
        #     slope_inside=0.05,
        #     slope_basin_break=2000,
        #     canyon_basin_break=2200,
        #     canyon_center=1000,
        #     canyon_half_width=100,
        # )

        create_init_flow_region(
            grid,
            initial_flow_concentration=1.0,
            initial_flow_thickness=h_ini,
            initial_region_radius=r_ini,
            initial_region_center=[1750, 300],
        )

        # making turbidity current object
        tc = TurbidityCurrent2D(
            grid,
            Cf=Cf_ini,
            alpha=0.4,
            kappa=0.05,
            nu_a=0.75,
            Ds=80 * 10**-6,
            h_init=0.0,
            Ch_w=10**(-5),
            R=1.0,
            h_w=0.01,
            C_init=0.0,
            implicit_num=100,
            implicit_threshold=1.0 * 10**-12,
            r0=1.5,
            water_entrainment=False,
            suspension=False,
            dflow=True,
            tan_delta=mu_ini,
        )

        # tc = TurbidityCurrent2D(grid,
        #                         Cf=0.004,
        #                         alpha=0.05,
        #                         kappa=0.25,
        #                         Ds=100 * 10**-6,
        #                         h_init=0.00001,
        #                         h_w=0.01,
        #                         C_init=0.00001,
        #                         implicit_num=20,
        #                         r0=1.5)

        return tc

    def run_flow(self, init_values):
        """ Run a flow to obtain the objective function
        """

        
        # Produce flow object
        tc = self.produce_flow(init_values[1], init_values[2], init_values[3], init_values[4])
        
        tc.eta = np.flip(tc.eta)

        # Run the model until endtime or 99% sediment settled
        # Ch_init = np.sum(tc.Ch)
        last = self.endtime
        dt = 50

        for i in tqdm(range(1, int(last/dt)+1), disable = False):
            tc.run_one_step(dt = dt)
            if np.amax(tc.U) < 0.1:
                break
            
        #tc.run_one_step(dt=self.endtime)
        # save_grid(
        #     tc.grid,
        #     'run-{0:.3f}-{1:.3f}-{2:.3f}.grid'.format(
        #         init_values[0], init_values[1], init_values[2]),
        #     clobber=True)

        bed_thick = tc.grid.node_vector_to_raster(
            tc.grid.at_node['flow__depth'])

        self.save_data(init_values, bed_thick)

        print('Run no. {} finished'.format(init_values[0]))

    def save_data(self, init_values, bed_thick_i):
        """Save result to a data file.
        """
        run_id = init_values[0]
        r_ini_i = init_values[1]
        h_ini_i = init_values[2]
        Cf_ini_i = init_values[3]
        mu_ini_i = init_values[4]

        dfile = netCDF4.Dataset(self.filename, 'a', share=True)
        r_ini = dfile.variables['r_ini']
        h_ini = dfile.variables['h_ini']
        Cf_ini = dfile.variables['Cf_ini']
        mu_ini = dfile.variables['mu_ini']
        bed_thick = dfile.variables['bed_thick']

        r_ini[run_id] = r_ini_i
        h_ini[run_id] = h_ini_i
        Cf_ini[run_id] = Cf_ini_i
        mu_ini[run_id] = mu_ini_i
        bed_thick[run_id, :, :] = bed_thick_i

        dfile.close()

    def run_multiple_flows(self):
        """run multiple flows
        """

        r_ini = self.r_ini
        h_ini = self.h_ini
        Cf_ini = self.Cf_ini
        mu_ini = self.mu_ini

        # Create list of initial values
        init_value_list = list()
        for i in range(len(r_ini)):
            init_value_list.append([i, r_ini[i], h_ini[i], Cf_ini[i], mu_ini[i]])

        # run flows using multiple processors
        pool = mp.Pool(self.processors)
        pool.map(self.run_flow, init_value_list)
        pool.close()
        pool.join()

    def create_datafile(self):

        num_runs = self.num_runs

        # check grid size
        tc = self.produce_flow(100, 100, 0.1, 0.1)
        grid_x = tc.grid.nodes.shape[0]
        grid_y = tc.grid.nodes.shape[1]
        dx = tc.grid.dx

        # record dataset in a netCDF4 file
        datafile = netCDF4.Dataset(self.filename, 'w')
        datafile.createDimension('run_no', num_runs)
        datafile.createDimension('grid_x', grid_x)
        datafile.createDimension('grid_y', grid_y)
        datafile.createDimension('basic_setting', 1)

        spacing = datafile.createVariable('spacing',
                                          np.dtype('float64').char,
                                          ('basic_setting'))
        spacing.long_name = 'Grid spacing'
        spacing.units = 'm'
        spacing[0] = dx

        r_ini = datafile.createVariable('r_ini',
                                        np.dtype('float64').char, ('run_no'))
        r_ini.long_name = 'Initial Radius'
        r_ini.units = 'm'
        h_ini = datafile.createVariable('h_ini',
                                        np.dtype('float64').char, ('run_no'))
        h_ini.long_name = 'Initial Height'
        h_ini.units = 'm'
        Cf_ini = datafile.createVariable('Cf_ini',
                                        np.dtype('float64').char, ('run_no'))
        Cf_ini.long_name = 'Friction coefficient'
        Cf_ini.units = '1'
        mu_ini = datafile.createVariable('mu_ini',
                                        np.dtype('float64').char, ('run_no'))
        mu_ini.long_name = 'Internal friction angle'
        mu_ini.units = '1'

        bed_thick = datafile.createVariable('bed_thick',
                                            np.dtype('float64').char,
                                            ('run_no', 'grid_x', 'grid_y'))
        bed_thick.long_name = 'Bed thickness'
        bed_thick.units = 'm'

        # close dateset
        datafile.close()


if __name__ == "__main__":
    # ipdb.set_trace()

    proc = 10  # number of processors to be used
    num_runs = 300
    Cmin, Cmax = [0.001, 0.03]
    rmin, rmax = [50., 200.]
    hmin, hmax = [25., 150.]

    C_ini = np.random.uniform(Cmin, Cmax, num_runs)
    r_ini = np.random.uniform(rmin, rmax, num_runs)
    h_ini = np.random.uniform(hmin, hmax, num_runs)

    rmf = RunMultiFlows(
        C_ini,
        r_ini,
        h_ini,
        'super191208_01.nc',
        processors=proc,
        endtime=4000.0,
    )
    rmf.create_datafile()
    start = time.time()
    rmf.run_multiple_flows()
    print("elapsed time: {} sec.".format(time.time() - start))
