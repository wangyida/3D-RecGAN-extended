import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import cm
from skimage.transform import resize
import argparse
from progressbar import ProgressBar


class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix='.jpg'):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath
            (excluding '.' and '..').
            filenames is a list of the names of the non-directory files
            in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list



class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix='.jpg'):
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        files_list = []

        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath
            (excluding '.' and '..').
            filenames is a list of the names of the non-directory files
            in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    special_file.endswith(self.postfix)
                    files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    special_file.startswith(self.prefix)
                    files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)


def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, normed=1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.hsv(c))

    plt.show()


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(
            np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
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


def plot_image(arr, name='depth.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_axis_off()
    arr = (arr-np.min(arr))/(np.max(arr) - np.min(arr)) * 255
    arr = np.uint8(arr)
    ax.set_axis_off()
    # ax.set_aspect('equal')

    plt.imshow(arr, cmap="hot")
    plt.savefig(name, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)


def plot_cube(cube, name='voxel.png', angle=140, IMG_DIM=80):
    from mpl_toolkits.mplot3d import Axes3D

    # cube = normalize(cube)
    facecolors = cm.gist_rainbow(normalize(cube))
    # make the alpha channel more similar to each others while 0 is still 0
    facecolors[:, :, :, -1] = 0.05 * np.tanh(cube * 1000)
    facecolors = explode(facecolors)
    filled = facecolors[:, :, :, -1] != 0

    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    # plot
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(20, angle)
    ax1.set_xlim(right=IMG_DIM*2)
    ax1.set_ylim(top=IMG_DIM*2)
    ax1.set_zlim(top=48*2)
    ax1.set_axis_off()
    ax1.voxels(x, y, z, filled, facecolors=facecolors)

    # plt.show()
    plt.savefig(name, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)

    """
    objects_name = ['empty', 'ceiling', 'floor', 'wall', 'window', 'chair', 'bed', 'sofa', 'table', 'tvs', 'furnture', 'object']
    for x in range(1, 11):
        fig = plt.figure(figsize=(30/2.54, 30/2.54))
        filled = explode(cube) == x
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.view_init(20, angle)
        ax1.set_xlim(right=IMG_DIM*2)
        ax1.set_ylim(top=IMG_DIM*2)
        ax1.set_zlim(top=48*2)
        ax1.set_title(objects_name[x])
        ax1.set_axis_off()
        ax1.voxels(x, y, z, filled, facecolors=facecolors)
        # plt.show()
        plt.savefig(name.replace('.png', '_'+objects_name[x]+'.png'), bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
    """


def plot_depvox(dir_dep, dir_vox, target_folder):
    label_start = dir_dep.rfind('/')+1
    label_end = dir_dep.find('.', label_start)
    arr = np.load(dir_dep)
    plot_image(
            arr,
            name=target_folder+'/depth/'+dir_dep[label_start:label_end]+'.png')

    arr = np.load(dir_vox)

    # ignore 255 and replace it with 0
    arr[arr == 255] = 0

    # show_histogram(arr)
    """
    transformed = np.clip(
            scale_by(np.clip(normalize(arr)-0.1, 0, 1)**0.4, 2)-0.1,
            0, 1)
    """
    resized = resize(arr, (48, 80, 80), mode='constant')
    plot_cube(
            np.rollaxis(resized[:, :, :], 2, 0),
            name=target_folder+'/voxel/'+dir_dep[label_start:label_end]+'.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument(
            '-d', action="store", dest="dir_dep",
            default="./SUNCGtrain_3001_5000", help='npy file for depth')
    parser.add_argument(
            '-v', action="store", dest="dir_vox",
            default="./SUNCGtrain_3001_5000", help='npy file for voxel')
    parser.add_argument(
            '-t', action="store", dest="target_folder",
            default="./target_folder", help='target folder for vis')
    parser.print_help()
    results = parser.parse_args()

    dir_dep = results.dir_dep
    dir_vox = results.dir_vox
    target_folder = results.target_folder
    scan = ScanFile(dir_dep)
    subdirs = scan.scan_subdir()
    files = scan.scan_files()

    """
    pbar = ProgressBar()
    for file_dep in pbar(files):
        file_vox = file_dep.replace(dir_dep, dir_vox, 1)
        plot_depvox(file_dep, file_vox, target_folder)
    """
    # vis for 3D FGAN
    pbar = ProgressBar()
    arr = np.load(results.dir_vox)
    for idx in pbar(range(0, arr.shape[0])):
        resized = resize(normalize(arr[idx, :, :, :]), (48, 80, 80), mode='constant')
        plot_cube(np.rollaxis(arr[idx, :, :, :], 2, 0), name = target_folder+'/'+str(idx)+'.png')
