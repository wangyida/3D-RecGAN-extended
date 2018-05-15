import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, normed=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()

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

def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr-mean)*fac + mean

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

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument('-s',
        action="store",
        dest="dir_src",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000",
        help='folder of paired depth and voxel')
    parser.print_help()
    results = parser.parse_args()

    arr = np.load(results.dir_src)
    show_histogram(arr)
    transformed = np.clip(scale_by(np.clip(normalize(arr)-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
    from skimage.transform import resize
    IMG_DIM = 50
    resized = resize(transformed, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')
    plot_cube(resized[:,::-1,:])
