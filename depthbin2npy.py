from struct import *
import numpy as np
# I considered using multiprocessing package, but I find this code version is fine.
# Welcome for your version with multiprocessing to make the reading faster.
# from joblib import Parallel, delayed
import multiprocessing
import time
from scipy import misc
import os
import argparse

def bin2array(file):
    start_time = time.time()
    with open(file,'r') as f:
            float_size = 4
            uint_size = 4
            total_count = 0
            cor = f.read(float_size*3)
            cors = unpack('fff', cor)
            # print("cors is {}",cors)
            cam = f.read(float_size*16)
            cams = unpack('ffffffffffffffff', cam)
            # print("cams %16f",cams)
            vox = f.read()
            numC = len(vox)/uint_size
            # print('numC is {}'.format(numC))
            checkVoxValIter = unpack('I'*numC, vox)
            checkVoxVal = checkVoxValIter[0::2]
            checkVoxIter = checkVoxValIter[1::2]
            checkVox = [i for (val, repeat) in zip(checkVoxVal,checkVoxIter) for i in np.tile(val, repeat)]
            # print('checkVox shape is {}'.format(len(checkVox)))
            checkVox = np.reshape(checkVox, (240,144,240))
    f.close()
    # print "reading voxel file takes {} mins".format((time.time()-start_time)/60)
    return checkVox

def png2array(file):
    image = misc.imread(file)
    return image

class ScanFile(object):
    def __init__(self,directory,prefix=None,postfix='.bin'):
        self.directory=directory
        self.prefix=prefix
        self.postfix=postfix

    def scan_files(self):
        files_list=[]

        for dirpath,dirnames,filenames in os.walk(self.directory):
            for special_file in filenames:
                if self.postfix:
                    if special_file.endswith(self.postfix):
                        files_list.append(os.path.join(dirpath,special_file))
                elif self.prefix:
                    if special_file.startswith(self.prefix):
                        files_list.append(os.path.join(dirpath,special_file))
                else:
                    files_list.append(os.path.join(dirpath,special_file))

        return files_list

    def scan_subdir(self):
        subdir_list=[]
        for dirpath,dirnames,files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Parser added')
    parser.add_argument('-s',
        action="store",
        dest="dir_src",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000",
        help='folder of paired depth and voxel')
    parser.add_argument('-t',
        action="store",
        dest="dir_tar",
        default="/media/wangyida/D0-P1/database/SUNCGtrain_3001_5000_devox",
        help='for storing generated npy')
    parser.print_help()
    results = parser.parse_args()

    # folder of paired depth and voxel
    dir_src=results.dir_src
    # for storing generated npy
    dir_tar=results.dir_tar
    
    # scan for depth files
    dir_depth = dir_tar + 'depth'
    scan_png = ScanFile(directory=dir_src, postfix='.png')
    files_png = scan_png.scan_files()
    
    # scan for semantic voxel files 
    dir_voxel = dir_tar + 'voxel'
    scan_bin = ScanFile(directory=dir_src, postfix='.bin')
    files_bin = scan_bin.scan_files()

    # making directories
    try:
        os.stat(dir_depth)
        os.stat(dir_voxel)
    except:
        os.mkdir(dir_depth) 
        os.mkdir(dir_voxel) 

    # save depth as npy files
    for file_png in files_png:
        depth = png2array(file=file_png)
        name_start = int(file_png.rfind('/'))
        name_end = int(file_png.find('.', name_start))
        np.save(dir_depth + file_png[name_start: name_end] + '.npy', depth)

    # save voxel as npy files
    for file_bin in files_bin:
        voxel = bin2array(file=file_bin)
        name_start = int(file_bin.rfind('/'))
        name_end = int(file_bin.find('.', name_start))
        np.save(dir_voxel + file_bin[name_start: name_end] + '.npy', voxel)
