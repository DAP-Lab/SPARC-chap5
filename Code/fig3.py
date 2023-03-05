import pandas as pd
import numpy as np

def annotateInteraction(axs, keywords, cs, interactionFile, startTime, duration):
    '''Adds interaction annotation to the axes given

    Parameters
        axs: list of axs to add annotation to
        keywords: keyword corresponding to each axis. If len(keywords) = len(axs) + 1, the last keyword is plotted in all axes
        cs: list of colours associated with each keyword
        interactionFile: path to csv file with the annotation of the interactions
        startTime: time to start reading the audio
        duration: length of audio to consider

    Returns
        axs: list of axes with annotation
    '''

    annotations = pd.read_csv(interactionFile, header=None)
    annotations.columns = ['Type', 'Start Time', 'End Time', 'Duration', 'Label']
    annotations = annotations.loc[((annotations['Start Time'] >= startTime) & (annotations['Start Time'] <= startTime+duration)) &
                                ((annotations['End Time'] >= startTime) & (annotations['End Time'] <= startTime+duration))
                                ]
    for i, keyword in enumerate(keywords):
        if i < len(axs):
            # keyword corresponds to a particular axis
            for _, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
                rand = np.random.random()# random vertical displacement for the label
                lims = axs[i].get_ylim()
                axs[i].annotate('', xy=(annotation['Start Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), xytext=(annotation['End Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec': cs[i], 'fc': cs[i]})
                axs[i].annotate(annotation['Label'], (annotation['Start Time']-startTime+annotation['Duration']/2, rand*(lims[1] - lims[0] - 100) + lims[0] + 150), ha='center')
        else:
            # keyword corresponds to all axes
            for ax in axs:
               for _, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
                rand = np.random.random()# random vertical displacement for the label
                ax.annotate('', xy=(annotation['Start Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), xytext=(annotation['End Time'] - startTime, rand*(lims[1] - lims[0] - 100) + lims[0] + 50), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec':cs[i], 'fc': cs[i]})
                ax.annotate(annotation['Label'], (annotation['Start Time']-startTime+annotation['Duration']/2, rand*(lims[1] - lims[0] - 100) + lims[0] + 150), ha='center') 
    return axs