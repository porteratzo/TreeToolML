#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# make angle circle
xval = np.arange(0, 2*np.pi, 0.01)
yval = np.ones_like(xval)

colormap = plt.get_cmap('hsv')
norm = mpl.colors.Normalize(0.0, 2*np.pi)

ax = plt.subplot(1, 1, 1, polar=True)
ax.scatter(xval, yval, c=xval, s=10000, cmap=colormap, norm=norm, linewidths=0)
ax.set_yticks([])
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib as mpl

# If displaying in a Jupyter notebook:
# %matplotlib inline 

# Generate a figure with a polar projection
fg = plt.figure(figsize=(8,8))
ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')

# Define colormap normalization for 0 to 2*pi
norm = mpl.colors.Normalize(0, 2*np.pi) 

# Plot a color mesh on the polar plot
# with the color set by the angle

n = 200  #the number of secants for the mesh
t = np.linspace(0,2*np.pi,n)   #theta values
r = np.linspace(.6,1,2)        #radius values change 0.6 to 0 for full circle
rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
colormap = plt.get_cmap('nipy_spectral')
c = tg                         #define color values as theta value
im = ax.pcolormesh(t, r, c.T,norm=norm, cmap=colormap)  #plot the colormesh on axis with colormap
ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
ax.tick_params(pad=15,labelsize=24)      #cosmetic changes to tick labels
ax.spines['polar'].set_visible(False)

# %%
