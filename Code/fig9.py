from matplotlib import pyplot as plt
import cv2
import numpy as np
import ffmpeg
import pandas as pd
from collections import defaultdict
import sys
import os
sys.path.append('../')
from utils import *

def drawHandTap(ax, handTaps, c='purple'):
    '''Plots the hand taps as vertical lines on the axis ax
    
    Parameters
        ax (plt.Axes.axis): axis to add hand taps to
        handTaps (np.array): array of hand tap timestamps
        c (str): color for plot
    '''
    for handTap in handTaps:
        ax.axvline(handTap, linestyle='--', c=c, alpha=0.6)
    return ax

def generateVideoWSquares(vid_path, timeStamps, dest_path='Data/Temp/vidWSquares.mp4', vid_size=(720, 576)):
    '''Function to genrate a video with rectangles for each tap
    
    Parameters
        vid_path (str): path to the original video
        timeStamps (list): list of time stamps with the following values for each time stamp - [time, keyword, [pos1, pos2], color]
        dest_path (str): file path to save video with clicks
        vid_size ((int, int)): video size to generate

    Returns
        None
    '''

    cap_vid = cv2.VideoCapture(vid_path)
    fps = cap_vid.get(cv2.CAP_PROP_FPS)
    framesToDraw = defaultdict(list)   # dictionary with frame numbers as keys and properties of square box to draw as list of values
    for timeRow in timeStamps:
        framesToDraw[int(np.around(timeRow[0]*fps))] = timeRow[1:]
    output = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*"XVID"), fps, vid_size)
    i = 0
    # generate video
    while(cap_vid.isOpened()):
        ret, frame = cap_vid.read()
        if ret == True:
            i+=1
            if i in framesToDraw.keys():
                frame = cv2.rectangle(frame, framesToDraw[i][1][0], framesToDraw[i][1][1], tuple([int(x) for x in framesToDraw[i][2]][::-1]), 3)    # converting color from BGR to RGB
            output.write(frame)
        else:
            # all frames are read
            break
    cap_vid.release()
    output.release()

def combineAudioVideo(vid_path='Data/Temp/vidWSquares.mp4', audio_path='audioWClicks.wav', dest_path='Data/Temp/FinalVid.mp4'):
    '''Function to combine audio and video into a single file

    Parameters
        vid_path (str): file path to the video file with squares
        audio_path (str): file path to the audio file with clicks
        dest_path (str): file path to store the combined file at

    Returns
        None

    '''
    
    vid_file = ffmpeg.input(vid_path)
    audio_file = ffmpeg.input(audio_path)
    (
        ffmpeg
        .concat(vid_file.video, audio_file.audio, v=1, a=1)
        .output(dest_path)
        .overwrite_output()
        .run()
    )
    print('Video saved at ' + dest_path)

def generateVideo(annotationFile, onsetKeywords, vidPath='Data/Temp/VS_Shree_1235_1321.mp4', tempFolder='Data/Temp/', pos=None, cs=None):
    '''Function to generate video with squares and clicks corresponding to hand taps
    
    Parameters
        annotationFile (str): file path to the annotation file with hand tap timestamps
        onsetKeywords (list): list of column names to read from annotationFile
        vidPath (str): file path to original file
        tempFolder (str): file path to temporary directory to store files in
        pos (list): list of [pos1, pos2] -> 2 opposite corners of the box for each keyword 
        cs (list): list of [R, G, B] colours used for each keyword
    
    Returns
        None
    '''
    annotations = pd.read_csv(annotationFile)
    timeStamps = []
    for i, keyword in enumerate(onsetKeywords):
        for timeVal in annotations[keyword].values[~np.isnan(annotations[keyword].values)]:
            timeStamps.append([timeVal, keyword, pos[i], cs[i]])
    timeStamps.sort(key=lambda x: x[0])

    # generate video 
    generateVideoWSquares(vid_path=vidPath, timeStamps=timeStamps, dest_path=os.path.join(tempFolder, 'vidWSquares.mp4'))

    # generate audio
    playAudioWClicks(audioPath=vidPath, onsetFile=annotationFile, onsetLabels=onsetKeywords, destPath=os.path.join(tempFolder, 'audioWClicks.wav'))

    # combine audio and video
    combineAudioVideo(vid_path=os.path.join(tempFolder, 'vidWSquares.mp4'), audio_path=os.path.join(tempFolder, 'audioWClicks.wav'), dest_path=os.path.join(tempFolder, 'finalVid.mp4'))