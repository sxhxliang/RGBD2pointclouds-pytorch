#!/usr/bin/python
# the resulting .ply file can be viewed for example with meshlab
# sudo apt-get install meshlab

"""
This script reads a registered pair of color and depth images and generates a
colored 3D point cloud in the PLY format.
"""

import argparse
import sys
import os
from PIL import Image
import numpy as np
import torch

import matplotlib.pyplot as plt


# 1242x375 kitti
focalLength = 718.856
centerX = 607.1928
centerY = 185.2157
scalingFactor = 1000


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # plt.imshow(depth_png)
    # plt.show()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float)# / 256.
    # depth[depth_png == 0] = -1.
    return depth

#def generate_pointcloud(rgb_file,depth_file,ply_file):
def generate_pointcloud():
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    
    rgb_img =  "0000000005.png"
    dep_img =  "0000000005d.png"
    master_dir = "./"
    rgb = Image.open(master_dir + rgb_img)
    dep = depth_read(master_dir + dep_img)

    dep = torch.from_numpy(dep)
    rgb = torch.from_numpy(np.array(rgb))

    points = []
    size = rgb.size()
    for v in range(size[1]):
        for u in range(size[0]):
            color = rgb[u,v,:]
            Z = dep[u, v] * 1.0/ scalingFactor
            if Z==0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(master_dir + "pc.ply","w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()

if __name__ == '__main__':
    """parser = argparse.ArgumentParser(description='''
    This script reads a registered pair of color and depth images and generates a colored 3D point cloud in the
    PLY format. 
    ''')
    parser.add_argument('rgb_file', help='input color image (format: png)')
    parser.add_argument('depth_file', help='input depth image (format: png)')
    parser.add_argument('ply_file', help='output PLY file (format: ply)')
    args = parser.parse_args()"""

    #generate_pointcloud(args.rgb_file,args.depth_file,args.ply_file)
    generate_pointcloud()
    

