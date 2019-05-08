#!/usr/bin/env python3
import os
import sys
import subprocess
import numpy as np
import pandas as pd
#import block_matching as bm
import vftk

def get_raw_perror_sequence(input_file):
    # This function will use h264poker to capture the prediction error sequence
    # of the video and return it as a list.
    perror_proc = subprocess.run(["h264poker", "-m", "perror", input_file, "-"],
        check=True, stdout=subprocess.PIPE)
    
    # Take byte array of output and decode into string.
    perror_str = perror_proc.stdout.decode("utf-8")
    # Strip out extra whitespace
    perror_str = perror_str.strip()    

    # split up string into useable chunks
    perror_list = perror_str.split(os.linesep)
    perror_list = perror_list[0:-1]
    
    # need to split each individual entry of the list. to get useable data
    perror_seq = []
    for index, item in enumerate(perror_list):
        tmp = item.strip().split(",")
        error = float(tmp[-1])
        perror_seq.append((index, error))

    return perror_seq

def get_inferred_perror(input_file, vftk):
    '''
    Calls a subprocess of vftk to perform prediction error inference.
    input_file (string) - Path to the input video file
    vftk (string) - Path to the vftk binary
    '''
    infer_proc = subprocess.run([vftk, 'perror', '-i', input_file],
            check=True, stdout=subprocess.PIPE)

    # Take byte array of output and decode into string.
    infer_str = infer_proc.stdout.decode("utf-8")
    infer_str = infer_str.strip()

    # Split up string into individual lines
    lines = infer_str.split(os.linesep)
    frame_nums = []
    error_1 = []
    error_2 = []
    error_3 = []

    for line in lines:
        tmp = line.strip().split(",")
        frame_nums.append(np.int32(tmp[0]))
        error_1.append(np.float64(tmp[1]))
        error_2.append(np.float64(tmp[2]))
        error_3.append(np.float64(tmp[3]))

    data = {'frame_num': frame_nums,
            1 : error_1,
            2 : error_2,
            3 : error_3}

    df = pd.DataFrame(data)
    perror_seq = []
    frame_nums = np.unique(df['frame_num'].to_numpy())
    for frame_num in frame_nums:
        frame_data = df[df['frame_num'] == frame_num].to_numpy()[:, 1:]
        min_match = np.nanargmin(frame_data, axis=1)
        matches = frame_data[np.arange(len(frame_data)), min_match]

        match_error = []
        for match in np.arange(3):
            if len(matches[min_match == match]) > 0:
                match_error.append(np.mean(matches[min_match == match]))
       
        perror_seq.append((frame_num, np.amax(np.array(match_error))))

    return perror_seq

def get_frame_types(input_file):
    # This function will use h264poker to captur the frame types of the input
    # video and return it as a dictionary, where the keys are the frame nums.
    frame_proc = subprocess.run(["h264poker", "-m", "ftypes", input_file, "-"],
        check=True, stdout=subprocess.PIPE)

    frame_str = frame_proc.stdout.decode("utf-8")
    frame_str = frame_str.strip()

    frame_list = frame_str.split(os.linesep)
    frame_list = frame_list[1:-1]

    ftype_seq = []
    for item in frame_list:
        tmp = item.strip().split(",")
        ftype = tmp[0]
        ftype_seq.append(ftype)

    return ftype_seq

def get_perror_sequence(input_file, vftk):
    # Function takes the raw prediction error sequence and cross checks the
    # frame types with that of the frame type sequence. Since we're only
    # interested in P-frames, B-frame data points will be skipped.
    raw_perror = get_inferred_perror(input_file, vftk) 
    ftype_seq = get_frame_types(input_file)
    perror_seq = []
    index = 0
    for item in raw_perror:
        # If ftype_seq[frame_index]
        if ftype_seq[item[0]] == 'P':
            perror_seq.append((index, item[0], item[1]))
            index += 1

    return np.array(perror_seq)

def save_perror_sequence(input_file, output_file, vftk):
    # Function saves the prediction error sequence to the file specified
    # and returns a numpy array containing the prediction error sequence
    perror_seq = get_perror_sequence(input_file, vftk)
    fmt = ['%d', '%d', '%f']
    np.savetxt(output_file, perror_seq, fmt=fmt, delimiter=',',
        header=input_file)

    return perror_seq

def load_perror_sequence(input_file):
    # Function loads the prediction error sequence from a text file
    # and returns a numpy array of containing the data stored in the file.
    converters = {0: lambda s: int(s.strip() or 0),
                  1: lambda s: int(s.strip() or 0),
                  2: lambda s: float(s.strip() or 0)}
    perror_seq = np.loadtxt(input_file, delimiter=',', converters=converters)
    return perror_seq
