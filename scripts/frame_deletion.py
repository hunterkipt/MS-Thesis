#!/usr/bin/env python3
from subprocess import run
import tempfile
import sys
import os

def delete_frames(video_file, start_frame, end_frame, output_file):
    # Create a temporary directory to hold video frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Dump the frames from the video file.
        dump_frames(video_file, tmp_dir)
        # Delete frames in the range [start_frame, end_frame]
        for filename in sorted(os.listdir(tmp_dir)):
            frame_num = int(filename[3:7])
            full_path = os.path.join(tmp_dir, filename)
            if frame_num >= start_frame and frame_num <= end_frame:
                os.remove(full_path)
            # Rename frames to make complete sequence
            elif frame_num >= end_frame:
                new_frame_num = frame_num - (end_frame - start_frame + 1)
                new_path = os.path.join(tmp_dir,
				"img{:04d}.png".format(new_frame_num))
                os.rename(full_path, new_path)
        # Images are now back in sequence. Reencode the video
        reencode_frames(tmp_dir, output_file)

def dump_frames(video_file, output_path):
    # This function will take in a video file path
    # and an output folder path, and dump the frames
    # from the input video to the output directory in the
    # form 'img0001.png, img0002.png, ..., imgNNNN.png'
    out_file = os.path.join(output_path, "img%04d.png")
    cmd = ["ffmpeg", "-y", "-r", "1", "-i", video_file, "-r", "1", out_file]
    try:
        run(cmd, check=True)
    except:
        print("FFMPEG Command Execution Failed.\n")

def reencode_frames(input_path, output_file):
    # This funcyion will take in a folder path full of images in the
    # form 'img0001.png, img0002.png, ..., imgNNNN.pnh' and reencode
    # them into a video specified by output_file
    in_file = os.path.join(input_path, "img%04d.png")
    cmd = ["ffmpeg","-y", "-r", "30", "-start_number", "1", "-i", in_file,
           "-c:v", "libx264","-x264opts", "keyint=30:min-keyint=30:scenecut=-1",
           "-bf", "0", "-b_strategy", "2", "-vf", "fps=30,format=yuv420p",
           output_file]

    try:
        run(cmd, check=False)
    except:
        print("FFMPEG Reencode Failed for" + output_file + "\n")


# ffmpeg -r 1 -i VID_20180717_070253.mp4 -r 1 "tmp/img%03d.png"

# ffmpeg -r 30 -start_number 0 -i ./tmp/img%03d.png -c:v libx264 -vf "fps=30,format=yuv420p" out.mp4

