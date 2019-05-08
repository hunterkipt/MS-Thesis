#!/usr/bin/env python3
import sys
import numpy as np
import h264pyker as h264
import contextlib
import diamond_search as ds
import multiprocessing as mp
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

@contextlib.contextmanager
def sane_open(filename=None):
    """
    Attempt to regain some well designed file output interface where I
    can open either stdout or a 'real' file.  Users specifying an output
    file of '-' for stdout will actually get expected behavior.
    """
    if filename and filename != '-':
        file_handle = open(filename, 'w')
    else:
        file_handle = sys.stdout
    try:
        yield file_handle
    finally:
        if file_handle is not sys.stdout:
            file_handle.close()

def block_matching(cur_frame, prev_frame, block_size=16, f_width=1920, 
                   f_height=1080, window_size=15):
    
    motion_vectors = [] 
    for x in np.arange(0, f_width + 1, block_size):
        for y in np.arange(0, f_height + 1, block_size):
            block_centroid = ds.get_block_centroid(x, y, block_size)
            mv = dict()
            mv['src'] = ds.diamond_search(cur_frame, prev_frame, block_centroid,
                                block_size, f_width, f_height, window_size)
            mv['dst'] = block_centroid
            motion_vectors.append(mv)
            # print("Matched Centroid: {}".format(str(block_centroid)))

    return motion_vectors

def count_valid_pixels(cur_frame):
    pixel_count = 0
    for mv in cur_frame.motionVectors():
        pixel_count += mv['size'][0] * mv['size'][1]

    return pixel_count

def mv_extract(infile, outfile, block_size=16):
    video = h264.PykerVideo(infile)
    f_width = video.width()
    f_height = video.height()
    if f_width % block_size != 0 or f_height % block_size != 0:
        print("Warning: Block size of {} is not evenly divisible for the given video.".format(block_size))

    with sane_open(outfile) as outfile_handle:
        frame = 0
        perror = []
        while video.seekNextFrame(): 
            cur_frame = video.getFrame()
            # If first frame, set prev_frame to cur_frame and continue
            if frame == 0:
                frame += 1
                perror.append((f_width*f_height, f_width*f_height))
                prev_frame = cur_frame
                continue
            
            #mvs = block_matching(cur_frame, prev_frame, block_size, 
            #                     f_width, f_height, window_size=45)
            mvs = cur_frame.motionVectors()
            # Create source and prediction frames
            Rs, Gs, Bs = prev_frame.imagePlanes()
            Rd, Gd, Bd = cur_frame.imagePlanes()
            Rp = np.zeros((f_height, f_width), dtype=np.uint8)
            Gp = np.zeros((f_height, f_width), dtype=np.uint8)
            Bp = np.zeros((f_height, f_width), dtype=np.uint8)
            for mv in mvs:
                break
                #print("{}: src: {} => dst: {}".format(frame, 
                #mv['src'], mv['dst']))
                src_x_min = mv['src'][0] - block_size//2
                src_x_max = mv['src'][0] + block_size//2
                src_y_min = mv['src'][1] - block_size//2
                src_y_max = mv['src'][1] + block_size//2
                dst_x_min = mv['dst'][0] - block_size//2
                dst_x_max = mv['dst'][0] + block_size//2
                dst_y_min = mv['dst'][1] - block_size//2
                dst_y_max = mv['dst'][1] + block_size//2
                
                Rp[dst_y_min:dst_y_max,:][:,dst_x_min:dst_x_max] = Rs[src_y_min:src_y_max,:][:,src_x_min:src_x_max]
                Gp[dst_y_min:dst_y_max,:][:,dst_x_min:dst_x_max] = Gs[src_y_min:src_y_max,:][:,src_x_min:src_x_max]
                Bp[dst_y_min:dst_y_max,:][:,dst_x_min:dst_x_max] = Bs[src_y_min:src_y_max,:][:,src_x_min:src_x_max]
                
            # Insert Visualization code here
            Re = np.abs(Rp - Rd)
            Ge = np.abs(Gp - Gd)
            Be = np.abs(Bp - Bd)

            valid_pixels = count_valid_pixels(cur_frame)
            
            # perror.append((frame, np.sum(np.vstack((Re,Ge,Be)))/valid_pixels))
            perror.append((f_width*f_height, valid_pixels))
            prev_frame = cur_frame
            frame += 1
            print("Matched Frame: {:d}".format(frame - 1))

    return perror
