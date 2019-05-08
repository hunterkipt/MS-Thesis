#!/usr/bin/env python3
#cython: language_level=3
import sys
import numpy as np
import h264pyker as h264
from cython.parallel import parallel, prange, threadid
cimport cython

def get_block_centroid(x, y, block_size=16):
    '''
    Gets macro-block centroid based on the block size.
    x (int) - the x coordinate of the top left point of the block
    y (int) - the y coordinate of the top left point of the block
    block_size (int) - the size of the macroblock (assumed square)
    Returns: (center_x, center_y) the center coordinates of the macroblock
    '''
    hop = block_size//2
    center_x = x + hop
    center_y = y + hop

    return center_x, center_y

def get_window_boundaries(block_centroid, window_size=15):
    '''
    Computes the window boundaries of the diamond search window
    for a given block centroid and window size
    '''
    if window_size % 2 == 1:
        window_min_x = block_centroid[0] - window_size//2
        window_min_y = block_centroid[1] - window_size//2
        window_max_x = block_centroid[0] + window_size//2
        window_max_y = block_centroid[1] + window_size//2

    else:
        window_min_x = block_centroid[0] - window_size//2
        window_min_y = block_centroid[1] - window_size//2
        window_max_x = block_centroid[0] + window_size//2 - 1
        window_max_y = block_centroid[1] + window_size//2 - 1
    
    return window_min_x, window_min_y, window_max_x, window_max_y

def setdiff2d(A, B):
    '''
    Calculates the set difference between A and B taking rows as elements
    A, B (array-like) - 2-d Matrices
    returns: diff - The set difference between A and B's rows
    '''
    A_rows = A.view([('', A.dtype)] * A.shape[1])
    B_rows = B.view([('', B.dtype)] * B.shape[1])
    
    diff = np.setdiff1d(A_rows, B_rows).view(A.dtype).reshape(-1, A.shape[1])
    return diff

def get_search_pattern(block_centroid, fmt="large"):
    '''
    Calculates the diamond search area given the block centroid
    '''
    # Define large diamond search pattern
    ldsp = np.array([[0, 0], [2, 0], [0, -2], [-2, 0], [0, 2],
                     [1, 1], [1, -1], [-1, -1], [-1, 1]])

    # Define small diamond search pattern
    sdsp = np.array([[0, 0], [1, 0], [0, -1], [-1, 0], [0, 1]])
  
    if fmt == "large":
        ldsp[:, 0] += block_centroid[0]
        ldsp[:, 1] += block_centroid[1]
        return ldsp

    else:
        sdsp[:, 0] += block_centroid[0]
        sdsp[:, 1] += block_centroid[1]
        return sdsp

cpdef int calc_diff(unsigned char a, unsigned char b) nogil:
    cdef int tmp_dif
    tmp_dif = b - a
    if tmp_dif < 0:
        tmp_dif *= -1

    return tmp_dif

@cython.boundscheck(False)
def calc_error(unsigned char[:, :] A, unsigned char[:, :] B):
    cdef Py_ssize_t x_max = A.shape[0]
    cdef Py_ssize_t y_max = A.shape[1]
    cdef Py_ssize_t x, y
    cdef int tmp_sum = 0

    for x in prange(x_max, nogil=True, schedule="dynamic"):
        for y in range(y_max):
            tmp_sum += calc_diff(A[x, y], B[x, y])


    #with ThreadPoolExecutor(max_workers=4) as exe:
    #    jobs = [exe.submit(calc_diff, B[x, y], A[x, y])
    #            for x in range(x_max) for y in range(y_max)]

    #for job in jobs:
    #    tmp_sum += job.result()
    return tmp_sum


cpdef block_similarity(A, B, A_centroid, B_centroid,
                     long block_size=16, long f_width=1920, long f_height=1080):
    '''
    Computes the block similarity between two blocks in a video frame
    A, B (PykerFrame) - The two frame objects to compare (same size assumed)
    A_centroid, B_centroid (tuple) - The x and y coordinates of the blocks
    block_size (int) - The size of a block (assumed square)
    returns: similarity - (float) a measure of the similarity between the blocks
    in frames A and B. None is returned if the centroids are outside the video
    frame.
    '''
    # Compute whether the proposed blocks are out of frame
    
    cdef Py_ssize_t A_x_min = (A_centroid[0] - block_size//2)
    cdef Py_ssize_t A_x_max = (A_centroid[0] + block_size//2)
    cdef Py_ssize_t A_y_min = (A_centroid[1] - block_size//2)
    cdef Py_ssize_t A_y_max = (A_centroid[1] + block_size//2)
    
    if A_x_min < 0 or A_y_min < 0 or A_x_max > f_width or A_y_max > f_height:
        return None
    
    cdef Py_ssize_t B_x_min = (B_centroid[0] - block_size//2)
    cdef Py_ssize_t B_x_max = (B_centroid[0] + block_size//2)
    cdef Py_ssize_t B_y_min = (B_centroid[1] - block_size//2)
    cdef Py_ssize_t B_y_max = (B_centroid[1] + block_size//2)
    
    if B_x_min < 0 or B_y_min < 0 or B_x_max > f_width or B_y_max > f_height:
        return None
    
    R_A, G_A, B_A = A.imagePlanes()
    R_B, G_B, B_B = B.imagePlanes()
    
    cdef unsigned char[:, :] R_As
    cdef unsigned char[:, :] G_As
    cdef unsigned char[:, :] B_As
    cdef unsigned char[:, :] R_Bs
    cdef unsigned char[:, :] G_Bs
    cdef unsigned char[:, :] B_Bs
    
    R_As = R_A[A_y_min:A_y_max,:][:,A_x_min:A_x_max]
    G_As = G_A[A_y_min:A_y_max,:][:,A_x_min:A_x_max]
    B_As = B_A[A_y_min:A_y_max,:][:,A_x_min:A_x_max]
    R_Bs = R_B[B_y_min:B_y_max,:][:,B_x_min:B_x_max]
    G_Bs = G_B[B_y_min:B_y_max,:][:,B_x_min:B_x_max]
    B_Bs = B_B[B_y_min:B_y_max,:][:,B_x_min:B_x_max]

    R_err = calc_error(R_As, R_Bs)#np.sum(np.abs(R_Bs - R_As))
    G_err = calc_error(G_As, G_Bs)#np.sum(np.abs(G_Bs - G_As))
    B_err = calc_error(B_As, B_Bs) #np.sum(np.abs(B_Bs - B_As))
    
    return np.mean((R_err, G_err, B_err))

def diamond_search(cur_frame, prev_frame, block_centroid, 
                   block_size=16, f_width=1920, f_height=1080, window_size=15):
    '''
    Computes the diamond_search algorithm for motion vector matching
    cur_frame (PykerFrame) - The current video frame object
    prev_frame (PykerFrame) - The previous video frame object
    block_centroid (tuple) - The x and y coordinates of the block centroid
    block_size (int) - how large the block is (assumed square).
    window_size (int) - the size of the search window (assumed square); The
    function will set the center of the search window at the centroid position
    of the block you are trying to match.
    Returns: (dx, dy) - The motion vector that gives the best match for the
    current block with one in the previous frame.
    '''
    # set up window boundaries
    window_min_x, window_min_y, window_max_x, window_max_y = get_window_boundaries(block_centroid, window_size)

    # Set initial search pattern
    fmt = "large"
    best_match = block_centroid
    searched = []

    done = False
    while not done:
        # Take set difference to find out which elements haven't been searched
        s_pattern = get_search_pattern(best_match, fmt=fmt)
        if len(searched) == 0:
            new_search = s_pattern
        else:
            new_search = setdiff2d(s_pattern, np.vstack(searched))
            new_search = np.vstack((s_pattern[0], new_search))
        
        similarity = []
        for point in new_search:
            # Compute block similarity
            score = block_similarity(cur_frame, prev_frame, block_centroid,
                                     point, block_size, f_width, f_height)
            # If point corresponds to a block outside the frame discount it
            if score is None:
                score = np.iinfo(np.int32).max
            
            # If point is outside window, discount it
            if (point[0] < window_min_x or point[0] > window_max_x or
               point[1] < window_min_y or point[1] > window_max_y):
                score = np.iinfo(np.int32).max

            similarity.append(score)
        
        # Take argmax of the similarity
        amin = np.argmin(np.array(similarity))
        # If small pattern, we're done
        if fmt == "small":
            done = True

        # If argmin is at the center of the pattern, switch to small search
        if amin == 0:
           fmt = "small"
        
        searched.append(new_search)
        best_match = new_search[amin]
    # Best match is now the centroid in the previous frame
    return best_match[0], best_match[1]

