"""
class for behavior data analysis recorded with YP's matlab code
GT 2021
"""

"""This is how the data looks in the matlabfile
# %init the data
# function initData()
# setappdata(0,'trackData',[]);       %track data = [time,xy,miceNum];
# setappdata(0,'ROIscoreData',[]);
# setappdata(0,'moveTag',[]);         %[time,activityState] 1=walk,0=rest over 1s(sampling period)
# setappdata(0,'checkFrames',[]);      %frames=[1,n/2,n] of episode (for post-check)
# setappdata(0,'newCheckTag',[1,1]);  %for myTimerCam function only
# %m=getappdata(0,'miceNum');
# bkgImg=getappdata(0,'bkImg2D');
# m=size(bkgImg,3);
# recTime=getappdata(0,'recTime'); ->length of the recording
# actHistPars=getappdata(0,'actHistPars');
# epn=ceil(recTime/actHistPars(1));
# bin=ceil(actHistPars(1)/actHistPars(2));
# setappdata(0,'actHistData',zeros(epn,bin,m));     %for all episodes (multi-files)
# setappdata(0,'actHistState',int8([]));      %state of actHistData,1=rest,0=awake
# setappdata(0,'sleepTag',zeros(3,m));   %curren state[number,duration(min),startpoint(min)]
# setappdata(0,'sleepData',zeros(1,3,m));       %[startpoint,endpoint,duration]=min
# setappdata(0,'velocityData',zeros(1,3,m));    %[time,speed,orietation]
# setappdata(0,'motionPIRData',zeros(1,2,m));     %[time,motionTag]
# setappdata(0,'actHistDataPIR',zeros(epn,bin,m));     %Data from PIR-motion sensor
# setappdata(0,'laserEvent',zeros(1,3));          %time points of laser stimulation: unit=minute [onset,tag,offset] 1=on;0=off
# setappdata(0,'SDEvent',0);     %time points of Sleep deprivation event (vibration+ultrasound), unit=minute
# setappdata(0,'ArduHistory',[]);
# setappdata(0,'Note',[]);		start and stop times of the fiber rec
# setappdata(0,'seizureEvent',-1);       %timepoints(sec) for seizure events
# global lastPos;
# lastPos=struct('Area',100,'Centroid',[1,1],'BoundingBox',[],'FilledImage',[],'Solidity',0.9);
# lastPos=repmat(lastPos,m,1);
#
# these are some comments (after %) for some key variables
"""

from os import chdir, listdir, walk, path

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import pandas as pd
# import scipy as sp
from scipy import io as sio
from scipy.signal import butter, filtfilt
from PIL import Image as im

from fnmatch import fnmatch
from datetime import datetime
from dateutil import parser

# loading and visualizig tracking data
# optimized for Jupyter notebook enviroment for now

def load_tracking(data_path):
	"""loads .mat files contains tracking info

	Parameters: 
	-----------
	data_path : string
		path to the data

	Returns:
	--------
	data dictionary"""

	data = sio.loadmat(data_path)
	return data

def show_background(data):
	"""plots backgroud image

	Parameters:
	-----------
	data : dictionary
		data dictinary generated by load_tracking

	Returns:
	--------
		matplotlib.image.AxesImage
	"""
	bk_data = im.fromarray(data['bkImg2D'])
	
	return imshow(bk_data, cmap='Greys_r')


def show_tracking(data):
	"""shows the tracking trace

	Parameters:
	-----------
	data : dictionary
		data dictinary generated by load_tracking
	"""
	track_data = data['trackData']

	# filtering out x=0, y=0 values
	
	arr = track_data
	arr[arr == 0]= 'nan'

	# plotting the data

	plt.plot(arr[0:,1], arr[0:,2])

def decimate_data(arr, N=10):
    """data downsampler

    Paramaters:
    -----------
    arr : numpy array
    N : int
        width of moving window for downsampling
        
    Returns:
    --------
    dictionary

    needs to be tested!!!
    """
    
    # decimatedData = {}
    decimatedSignal = []
    # decimatedTime = []

    for i in range(0, len(arr), N):        
            # This is the moving window mean
            mean_wnd = np.mean(arr[i:i+N-1])
            decimatedSignal.append(mean_wnd)
    # np.array(decimatedSignal)

    # time_x = range(len(arr))
    # time_x = time_x[::N] # go from beginning to end of array in steps on N
    # time_x = time_x[:len(arr)]

    # decimatedData['decimatedSignal'] = decimatedSignal
    # decimatedData['decimatedTime'] = time_x
    
    return np.array(decimatedSignal)

def get_imaging_folders(data_path, foldernames):
	"""
	grabs imaging folders with similar names

	Paramters:
	----------
	data_path : str
		path like str
	foldername : str
		common pattern in the folder

	Retunrs:
	--------
	list of imaging folders, sorted"""

	im_folders = []

	for folder in listdir(data_path):
	    if fnmatch(folder, foldernames+'*'):
	    	im_folders.append(folder)
	print(f"there are {len(im_folders)} imaging folders")
		
	return sorted(im_folders)

def abs_start_time(beh_file_path):
	"""retunrs the start time of the tracking data

	Parameters:
	-----------
	beh_file_path : str
		pointer to the location of the file
	Returns:
	--------
	datetime object
	"""

	(head, tail) = path.split(beh_file_path)
	time_stamp = tail.split('.')
	start_timestamp = datetime.strptime(time_stamp[0],'%Y-%m-%d %H-%M-%S')

	return start_timestamp

def behavior_df_generator(beh_data):
    """Builds a dataframe from behavior data
        
    """
    
    tracking_df = pd.DataFrame(columns=['tracking_time', 'tracking_x',
                                       'tracking_y', 'velocity_speed'])


    tracking_df['tracking_time'] = beh_data['trackData'][0:,0]
    tracking_df['tracking_x'] = beh_data['trackData'][0:,1]
    tracking_df['tracking_y'] = beh_data['trackData'][0:,2]
    # tracking_df['velocity_time'] = beh_data['velocityData'][0:,0]
    tracking_df['velocity_speed'] = beh_data['velocityData'][0:,1]
    tracking_df['file'] = beh_data['filename'][0].split("\\")[3]
    tracking_df['start_time'] = beh.abs_start_time(beh_file)
    
    return tracking_df