import numpy as np
import frame_deletion
import prediction_error
import os
from tqdm import tqdm
#V_19700116_170516.mp4
#V_19700116_170527.mp4
#V_19700116_170612.mp4
#V_19700116_170642.mp4

source_dir = "../videos/KodakEktra"
dest_dir = source_dir + "_fdel"
start_frame = 1
end_frame = 15
vftk = "./vftk/build/bin/vftk"

# If Destination directory doesn't exist, make it
if not (os.path.isdir(dest_dir)):
    os.mkdir(dest_dir)

# Create folders for prediction error sequences in each directory
source_perror_dir = os.path.join(source_dir, "perror_infer_new")
dest_perror_dir = os.path.join(dest_dir, "perror_infer_new")

if not (os.path.isdir(source_perror_dir)):
    os.mkdir(source_perror_dir)

if not (os.path.isdir(dest_perror_dir)):
    os.mkdir(dest_perror_dir)

for filename in tqdm(sorted(os.listdir(source_dir))):
    # Set up file basenames for data saving
    basename = filename.split(".")[0]
    new_basename = basename + "_" + str(start_frame) + "_" +  str(end_frame)
    source_perror = os.path.join(source_perror_dir, basename + ".txt")
    source_video = os.path.join(source_dir, filename)
    dest_perror = os.path.join(dest_perror_dir, new_basename + ".txt")
    dest_video = os.path.join(dest_dir, new_basename + ".mp4")
    
    # Delete video frames from source and save to destination
    frame_deletion.delete_frames(source_video, start_frame, end_frame,
        dest_video)

    
    # Save prediction error sequences
    prediction_error.save_perror_sequence(source_video, source_perror, vftk)
    prediction_error.save_perror_sequence(dest_video, dest_perror, vftk)

