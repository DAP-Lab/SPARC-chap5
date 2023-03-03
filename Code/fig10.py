import math
import parselmouth
import seaborn as sns
import pandas as pd
import librosa
import numpy as np
import sys
sys.path.append('../')
from utils import drawAnnotation

def intensityContour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, timeStep=0.01, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', xticks=False, yticks=False, cAnnot='red', annotLabel=True, annotAlpha=0.8):
    '''Calculates the intensity contour for an audio clip

    Parameters
        audio (np.array): loaded audio time series
        sr (int): sample rate of audio time series/ to load the audio at
        audioPath (str): path to audio file; only needed if audio is None
        startTime (float): time to start reading audio file
        duration (float): duration of the audio file to read
        minPitch (float): minimum pitch to read for contour extraction
        timeStep (float): time steps in which audio is extracted
        ax (plt.Axes.axis): axis to plot the pitch contour in
        freqXlabels (int): time (in seconds) after which each x label occurs
        annotate (bool): if True, will annotate tala markings
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetTimeKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        xticks (bool): if True, will plot xticklabels
        yticks (bool): if True, will plot yticklabels
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        cAnnot (str): color of the annotation
        annotAlpha: controls opacity of the annotation lines

    Returns:
        ax: plot of pitch contour if ax was not None
    
    '''
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
    if audio is None:
        # if audio is not given, load audio from audioPath
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration = duration)
    snd = parselmouth.Sound(audio, sr)
    intensity = snd.to_intensity(time_step=timeStep, minimum_pitch=minPitch)
    
    # if ax is None, raise error
    if ax is None:
        Exception('ax parameter has to be provided')
    # plot the contour
    return plotIntensity(intensity=intensity, ax=ax, startTime=startTime, duration=duration, freqXlabels=freqXlabels, xticks=xticks, yticks=yticks, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, cAnnot=cAnnot, annotLabel=annotLabel, annotAlpha=annotAlpha)

def plotIntensity(intensity=None, ax=None, startTime=0, duration=None, freqXlabels=5, xticks=True, yticks=True, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='red', annotLabel=True, annotAlpha=0.8):
    '''Function to plot a computed intensity contour

    Parameters
        intensity (parselmouth.intensity): intensity object from `intensityContour`
        ax (plt.Axes.axis): axis object on which plot is to be plotted
        startTime (float): start time for x labels in the plot
        duration (float): duration of audio in the plot (used for x labels)
        freqXlabels (int): time (in seconds) after which each x label occurs
        xticks (bool): if True, will print x tick labels
        yticks (bool): if True, will print y tick labels
        annotate (bool): if True will mark annotations provided
        cyclePath (str): path to file with cycle annotations; used only if annotate is True
        numDiv (int): number of divisions to add between each marked cycle; used only if annotate is True
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot (str): colour to draw annotation lines in; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True
        annotAlpha (float): controls opacity of the line drawn
    Returns
        ax: plotted axis
    
    '''
    if intensity is None:
        Exception('No intensity contour provided')
    if ax is None:
        Exception('ax parameter has to be provided')
    yvals = intensity.values[0]
    xvals = intensity.xs()
    ax = sns.lineplot(x=xvals, y=yvals, ax=ax, color=cAnnot);
    ax.set(xlabel='Time Stamp (s)' if xticks else '', 
    ylabel='Intensity (dB)' if yticks else '', 
    title='Intensity Contour', 
    xlim=(0, duration), 
    xticks=(np.arange(0, duration, freqXlabels)), 
    xticklabels=(np.arange(startTime, duration+startTime, freqXlabels) )if xticks else [])
    if not yticks:
        ax.set(yticklabels=[])
    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel, annotAlpha=annotAlpha)
    return ax

def plot_hand(annotationFile=None, startTime=0, duration=None, freqXLabels=5, vidFps=25, ax=None, annotate=False, cyclePath=None, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', numDiv=0, cAnnot='yellow', annotLabel=False, annotAlpha=0.8, xticks=False, yticks=False, vidOffset=0, lWristCol='LWrist', rWristCol='RWrist', wristAxis='y'):
    '''Function to show hand movement
    
    Parameters
        annotationFile (str): file path to openpose annotations
        startTime (float): start time for x labels in the plot
        duration (float): duration of audio in the plot (used for x labels)
        freqXlabels (int): time (in seconds) after which each x label occurs
        vidFps (float): fps of the video
        ax (plt.Axes.axis): axis object on which plot is to be plotted
        annotate (bool): if True will mark annotations provided
        cyclePath (str): path to file with cycle annotations; used only if annotate is True
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        numDiv (int): number of divisions to add between each marked cycle; used only if annotate is True
        cAnnot (str): colour to draw annotation lines in; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True
        annotAlpha (float): controls opacity of the line drawn
        xticks (bool): if True, will print x tick labels
        yticks (bool): if True, will print y tick labels
        videoOffset (float): number of seconds offset between video and audio; time in audio + videioOffset = time in video
        lWristCol (str): name of the column with left wrist data in annotationFile
        rWristCol (str): name of the column with right wrist data in annotationFile
        wristAxis (str): Level 2 header denoting axis along which movement is plotted
        
    Returns
        ax: plotted axis

    '''
    startTime = math.floor(startTime) + np.around(vidOffset)   # set start time to an integer, for better readability on the x axis of the plot
    duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
    movements = pd.read_csv(annotationFile, header=[0, 1])
    lWrist = movements[lWristCol][wristAxis].values[startTime*vidFps:(startTime+duration)*vidFps]
    rWrist = movements[rWristCol][wristAxis].values[startTime*vidFps:(startTime+duration)*vidFps]
    xvals = np.linspace(0, duration, vidFps*duration, endpoint=False)
    ax.plot(xvals, lWrist, label='Left Wrist')
    ax.plot(xvals, rWrist, label='Right Wrist')
    ax.set(xlabel='Time Stamp (s)' if xticks else '', 
    ylabel='Wrist Position', 
    title='Wrist Position Vs. Time', 
    xlim=(0, duration), 
    xticks=(np.arange(0, duration, freqXLabels)), 
    xticklabels=(np.arange(startTime, duration+startTime, freqXLabels) )if xticks else [],
    )
    if not yticks:
        ax.set(yticklabels=[])
    ax.invert_yaxis()
    ax.legend()
    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime-np.around(vidOffset), duration, ax, c=cAnnot, annotLabel=annotLabel)
    return ax