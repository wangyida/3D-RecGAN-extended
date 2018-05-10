import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, angle=320):
    cube = normalize(cube)
    
    facecolors = cm.viridis(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM*2)
    ax.set_ylim(top=IMG_DIM*2)
    ax.set_zlim(top=IMG_DIM*2)
    ax.set_axis_off()
    
    ax.voxels(x, y, z, filled, facecolors=facecolors)
    plt.show()

# A few notes on this implementation:
# 
#  * Instead of colors as strings, I'm using a 4D colors array, where the last dimension (of size 4) holds the red, green, blue, and alpha (transparency) values. Doing `facecolors[:,:,:,-1] = cube` makes the alpha equal to the voxel value.
#  * I'm still using Viridis, the default color map. You can use [any map you like](https://matplotlib.org/users/colormaps.html) that's supported by matplotlib, by changing the call to `cm.viridis`.
#  * I'm setting some axis limits to make sure that all the plots are on the same scales, even if I truncate the image to show a cross-section.
#  * You can add a call to `ax.set_axis_off()` if you want to remove the background and axis ticks.
# 
# Oh, and if you were wondering, this is where `explode` handling 4D arrays comes in handy.
# 
# 
# ## Results
# 
# So first, a cut view of the skull:

# In[ ]:


plot_cube(resized[:35,::-1,:25])


# (I'm plotting the y-axis backwards so that the eyes are in front).
# 
# A view from the back, cutting through in diagonal:

# In[ ]:


cube = np.copy(resized)

for x in range(0, IMG_DIM):
    for y in range(0, IMG_DIM):
        for z in range(max(x-y+5, 0), IMG_DIM):
            cube[x, y, z] = 0
plot_cube(cube, angle=200)


# And a full view, where you can see the nasal cavity and make out the eye sockets at the very bottom:

IMG_DIM = 50
plot_cube(checkVox[:,::-1,:])


# I hope you enjoyed this tutorial! Feel free to drop me suggestions for improvements, questions, or other random notes below. You can also look at [`voxel`'s documentation](https://matplotlib.org/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D.voxels) for more details.
# 
# Here are the library versions I've used for this tutorial:
# 
#     matplotlib==2.1.0
#     nibabel==2.2.1
#     numpy==1.13.3
#     requests==2.18.4
#     scikit-image==0.13.1
# 
# You can also [download the notebook](/content-static/matplotlib-3d/matplotlib-3d.ipynb).
# 
# [^1]: Büchel, Christian, and K. J. Friston. "Modulation of connectivity in visual pathways by attention: cortical interactions evaluated with structural equation modelling and fMRI." _Cerebral cortex (New York, NY: 1991)_ 7.8 (1997): 768-778.
