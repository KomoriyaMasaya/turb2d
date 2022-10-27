import netCDF4
import numpy as np
import inspect
import matplotlib.pyplot as plt
import matplotlib.cm as cm

infile = "tc0001.nc"

nc = netCDF4.Dataset(infile,'r')

for x in inspect.getmembers(nc):
  print (x)

val = nc.variables["flow__depth"][:,:,:]
print(val)

max = np.max(val)
print('max : {0}'.format(max))

min = np.min(val)
print('min : {0}'.format(val))

sum = np.sum(val)
print('sum :{0}'.format(sum))

#nc.variables["flow__vertical_velocity_at_node"][:,:,:]
#print(nc.variables["flow__vertical_velocity_at_node"][:,:,:])

#np.max(nc.variables["flow__vertical_velocity_at_node"][:,:,:])
#print('vmax : {0}'.format(np.max(nc.variables["flow__vertical_velocity_at_node"][:,:,:])))
#
#np.min(nc.variables["flow__vertical_velocity_at_node"][:,:,:])
#print('vmin : {0}'.format(np.min(nc.variables["flow__vertical_velocity_at_node"][:,:,:])))

#print ('dim = 'format(nc.dimensions))
#print (nc.variables)

#data = nc.variables["flow__horizontal_velocity_at_node"][:,:,:]
val = np.squeeze(val)
#print(data.shape)

plt.imshow(val)
plt.title("Plot 2D array")
plt.savefig("hoge.png")
