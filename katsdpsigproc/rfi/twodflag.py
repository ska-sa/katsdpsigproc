#Library to contain RFI flagging routines and other RFI related functions
import katdal
import katpoint 
import warnings
warnings.simplefilter('ignore')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from mpl_toolkits.axes_grid import Grid

import matplotlib.pyplot as plt #; plt.ioff()
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import matplotlib

import numpy as np
import optparse
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import skimage
import skimage.morphology
import math

import pickle

import h5py
import os
import shutil
import multiprocessing as mp
import time
import itertools

#Supress warnings
#import warnings
#warnings.simplefilter('ignore')

def running_mean(x, N, axis=None):
    #Fast implementation of a running mean (array x with width N)
    #Stolen from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    #And modified to allow axis selection
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=axis), axis=axis)
    return np.apply_along_axis(lambda x: (x[N:] - x[:-N])/N, axis, cumsum) if axis else (cumsum[N:] - cumsum[:-N])/N


def getbackground(data,in_flags=None,iterations=3,spike_width_time=10,spike_width_freq=10,reject_threshold=2.0,interp_nonfinite=True):
    """Determine a smooth background through a 2d data array by iteratively smoothing
    the data with a gaussian
    """
    #Make mask array
    mask=np.ones(data.shape,dtype=np.float)
    #Mask input flags if provided
    if in_flags is not None:
        mask[in_flags]=0.0
    #Filter Brightest spikes
    for i in range(2):
        median = np.nanmedian(data[~in_flags])
        mask[data-median > reject_threshold*3*np.nanstd(data[~in_flags])]=0.0
    #Next convolve with Gaussians with increasing width from iterations*spike_width to 1*spike_width
    for extend_factor in range(iterations,0,-1):
        #Convolution sigma
        sigma=np.array([min(spike_width_time*extend_factor,max(data.shape[0]//10,1)),min(spike_width_freq*extend_factor,data.shape[1]//10)])
        #sigma=np.array([1,min(spike_width_freq*extend_factor,data.shape[1]//10)])
        #Get weight and background convolved in time axis
        weight=ndimage.gaussian_filter(mask,sigma,mode='constant',cval=0.0,truncate=3.0)
        #Smooth background and apply weight
        background=ndimage.gaussian_filter(data*mask,sigma,mode='constant',cval=0.0,truncate=3.0)/weight
        residual=data-background
        #Reject outliers
        residual=residual-np.median(residual[np.where(mask)])
        mask[np.abs(residual)>reject_threshold*np.nanstd(residual[np.where(mask)])]=0.0
    weight = ndimage.gaussian_filter(mask,sigma,mode='constant',cval=0.0,truncate=3.0)
    background = ndimage.gaussian_filter(data*mask,sigma,mode='constant',cval=0.0,truncate=3.0)/weight

    def linearly_interpolate_nans(y):
        # Create X matrix for linreg with an intercept and an index
        nan_locs = np.isnan(y)
        
        if np.all(nan_locs):
            y[:] = 0
        else:
            X = np.nonzero(~nan_locs)[0]
            Y = y[X]
            y[nan_locs] = np.interp(np.nonzero(nan_locs),X,Y)[0]
        return y

    background = np.apply_along_axis(linearly_interpolate_nans, 1, background)
    return background


def get_scan_flags(flagger,data,flags):
    """
    Function to run the flagging for a single scan. This is used by multiprocessing
    to avoid pickling problems therein. It can also be used to run the flagger
    independently of multiprocessing.
    Inputs:
        flagger: A flagger object with a method for running the flagging
        data: a 2d array of data to flag
        flags: a 2d array of prior flags
    Outputs:
        a 2d array of derived flags for the scan
    """
    return flagger._detect_spikes_sumthreshold(data,flags)


class sumthreshold_flagger():
    def __init__(self,background_iterations=1, spike_width_time=10, spike_width_freq=10, outlier_sigma_freq=4.5, outlier_sigma_time=5.5, 
                background_reject=2.0, num_windows=5, average_time=1, average_freq=1, time_extend=3, freq_extend=3, freq_chunks=7, debug=False):
        self.background_iterations=background_iterations
        spike_width_time/=average_time
        spike_width_freq/=average_freq
        self.spike_width_time=spike_width_time
        self.spike_width_freq=spike_width_freq
        self.outlier_sigma_freq=outlier_sigma_freq
        self.outlier_sigma_time=outlier_sigma_time
        self.background_reject=background_reject
        # Range of windows from 1 to 2*spike_width
        self.window_size_freq=np.unique(np.logspace(0,max(0,np.log10(spike_width_freq + 1e-6)),num_windows,endpoint=True,dtype=np.int))
        self.window_size_time=np.unique(np.logspace(0,max(0,np.log10(spike_width_time + 1e-6)),num_windows,endpoint=True,dtype=np.int))
        self.average_time=average_time
        self.average_freq=average_freq
        self.debug=debug
        #Internal parameters
        #Fraction of data flagged to extend flag to all data
        self.flag_all_time_frac = 0.6
        self.flag_all_freq_frac = 0.8
        #Extend size of flags in time and frequency
        self.time_extend = time_extend
        self.freq_extend = freq_extend
        #Set up subbands in frequency
        #Can be an int to equally subdivide into the required chunks, or a list which specifies the channels of the chunk edges.
        self.freq_chunks = np.array(freq_chunks)//average_freq
        #Falloff exponent for sumthreshold
        self.rho = 1.3
        if debug:
            print 'Initialise flagger:'
            print 'spike_width_time,spike_width_freq:', spike_width_time,spike_width_freq
            print 'window_size_time,window_size_freq:',self.window_size_time,self.window_size_freq
            print 'Frequency splitting channels:',self.freq_chunks

    def get_flags(self,data,flags=None,num_cores=8):
        if self.debug: start_time=time.time()
        if flags is None:
            in_flags = np.repeat(None,in_data.shape[0]).reshape((in_data.shape[0]))
        out_flags=np.empty(data.shape,dtype=np.bool)
        async_results=[]
        p=mp.Pool(num_cores)
        for i in range(data.shape[-1]):
            async_results.append(p.apply_async(get_scan_flags,(self,data[...,i],flags[...,i],)))
        p.close()
        p.join()
        #for i in range(data.shape[-1]):
        #    get_scan_flags(self,data[...,i],flags[...,i])
        for i,result in enumerate(async_results):
            out_flags[...,i]=result.get()     
        if self.debug: 
            end_time=time.time()
            print "TOTAL SCAN TIME: %f"%((end_time-start_time)/60.0)
        return out_flags

    def _average(self,data,flags):
        #Only works if self.average_time and self.average_freq divide into data.shape
        new_time_axis = data.shape[0]//self.average_time
        new_freq_axis = data.shape[1]//self.average_freq
        bin_area = self.average_time*self.average_freq
        avg_data = data.reshape(new_time_axis,self.average_time,new_freq_axis,self.average_freq)
        avg_flags = flags.reshape(new_time_axis,self.average_time,new_freq_axis,self.average_freq)
        avg_data = np.nansum(np.nansum(avg_data*(~avg_flags),axis=3),axis=1)
        avg_flags = np.nansum(np.nansum(avg_flags,axis=3),axis=1)
        bin_weight = (bin_area - avg_flags)
        avg_flags = (avg_flags == bin_area)
        #Avoid NaNs where all data is flagged (data will become zero)
        bin_weight[avg_flags==True] = 1
        avg_data /= bin_weight
        return avg_data, avg_flags

    def _detect_spikes_sumthreshold(self,in_data,in_flags):
        if self.debug: 
            start_time=time.time()
            back_time=0.
            st_time=0.
        #Create flags array
        if in_flags is None:
            in_flags = np.zeros(in_data.shape, dtype=np.bool)
        if self.average_time > 1 or self.average_freq > 1:
            in_data, in_flags = self._average(in_data,in_flags)
        out_flags = np.zeros(in_data.shape, dtype=np.bool)
        #Set up chunks
        if type(self.freq_chunks) is int:
            freq_chunks = np.linspace(0,in_data.shape[1],self.freq_chunks+1,dtype=np.int)
        else:
            #Assume it's an iterable and correct for averaging
            freq_chunks = self.freq_chunks
            #Append the start and end of the channel range if not specified in input list
            freq_chunks = np.unique(np.append(freq_chunks,[0,in_data.shape[1]]))
            freq_chunks.sort()
        #Number of channels to pad start and end of each chunk (factor of 3 is gaussian smoothing box size in getbackground)
        freq_chunk_overlap = int(self.spike_width_freq)*3
        #Loop over chunks
        for chunk_num in range(len(freq_chunks)-1):
            if self.debug: back_start=time.time()
            #Chunk the input and create output chunk
            chunk_start = freq_chunks[chunk_num]
            chunk_end = freq_chunks[chunk_num+1]
            chunk = slice(chunk_start-freq_chunk_overlap,chunk_end+freq_chunk_overlap) if chunk_num > 0 \
                                                else slice(chunk_start,chunk_end+freq_chunk_overlap)
            in_data_chunk = in_data[:,chunk]
            #Copy in_flags- as we'll be changing them.
            in_flags_chunk = in_flags[:,chunk]
            out_flags_chunk = np.zeros_like(in_flags_chunk,dtype=np.bool)
            #Flag a 1d median spectrum in frequency
            spec_flags = np.all(in_flags_chunk,axis=0).reshape((1,-1,))
            #Un-flag channels that are completely flagged in time to avoid nans in the median
            in_flags_chunk[:,spec_flags[0]]=False
            spec_data = np.nanmedian(np.where(in_flags_chunk,np.nan,in_data_chunk),axis=0).reshape((1,-1,))
            #Re-flag channels that are completely flagged in time.
            in_flags_chunk[:,spec_flags[0]]=True
            spec_background = getbackground(spec_data,in_flags=spec_flags,iterations=self.background_iterations,spike_width_time=self.spike_width_time,\
                                                spike_width_freq=self.spike_width_freq,reject_threshold=self.background_reject,interp_nonfinite=True)
            av_dev = spec_data - spec_background
            spec_flags = self._sumthreshold(av_dev,spec_flags,1,self.window_size_freq,self.outlier_sigma_freq)
            #Extend spec_flags to number of timestamps
            out_flags_chunk[:] = spec_flags
            #Use the spec flags in the mask
            in_flags_chunk |= out_flags_chunk
            #Flag the 2d data in a time,frequency view
            background_chunk = getbackground(in_data_chunk,in_flags=in_flags_chunk,iterations=self.background_iterations,spike_width_time=self.spike_width_time,\
                                                spike_width_freq=self.spike_width_freq,reject_threshold=self.background_reject,interp_nonfinite=True)
            if self.debug:
                back_time += (time.time() - back_start)
                st_start = time.time()
            #Subtract background
            av_dev = in_data_chunk - background_chunk
            #Sumthershold along time axis
            time_flags_chunk = self._sumthreshold(av_dev,in_flags_chunk,0,self.window_size_time,self.outlier_sigma_time)
            #Sumthreshold along frequency axis- with time flags in the mask
            freq_flags_chunk = self._sumthreshold(av_dev,in_flags_chunk | time_flags_chunk,1,self.window_size_freq,self.outlier_sigma_freq)
            #Combine all the flags
            out_flags_chunk |= time_flags_chunk | freq_flags_chunk
            out_flags[:,chunk_start:chunk_end] = out_flags_chunk[:,freq_chunk_overlap:freq_chunk_overlap+chunk_end-chunk_start] if chunk_num > 0 \
                                                                    else out_flags_chunk[:,0:chunk_end-chunk_start]
            if self.debug: st_time += time.time()-st_start
        if self.average_freq > 1:
            out_flags=np.repeat(out_flags,self.average_freq,axis=1)
        if self.average_time > 1:
            out_flags=np.repeat(out_flags,self.average_time,axis=0)
        #Extend flags in freq and time (take into account averaging)
        if self.freq_extend > 1:
            out_flags = ndimage.convolve1d(out_flags, [True]*self.freq_extend, axis=1, mode='reflect')
        if self.time_extend > 1:
            out_flags = ndimage.convolve1d(out_flags, [True]*self.time_extend, axis=0, mode='reflect')
        #Flag all freqencies and times if too much is flagged.
        out_flags[:,np.where(np.sum(out_flags,dtype=np.float,axis=0)/out_flags.shape[0] > self.flag_all_time_frac)[0]]=True
        out_flags[np.where(np.sum(out_flags,dtype=np.float,axis=1)/out_flags.shape[1] > self.flag_all_freq_frac)]=True
        if self.debug:
            end_time=time.time()
            print "Shape %d x %d, BG Time %f, ST Time %f, Tot Time %f"%(in_data.shape[0], in_data.shape[1], back_time, st_time, end_time-start_time)
        return out_flags


    def _sumthreshold(self,input_data,flags,axis,window_bl,sigma):

        sd_mask = (input_data==0.) | ~np.isfinite(input_data) | flags
        #Get standard deviations along the axis using MAD
        estm_stdev = 1.4826 * np.nanmedian(np.abs(np.where(sd_mask,np.nan,input_data)),axis=axis)
        # Identify initial outliers (again based on normal assumption), and replace them with local median
        threshold = sigma * estm_stdev
        output_flags = np.zeros_like(flags,dtype=np.bool)
        for window in window_bl:
            if window>input_data.shape[axis]: break
            #The threshold for this iteration is calculated from the initial threshold
            #using the equation from Offringa (2010).
            thisthreshold = np.expand_dims(threshold / pow(self.rho,(math.log(window)/math.log(2.0))), axis).repeat(input_data.shape[axis],axis=axis)
            #Set already flagged values to be the value of this threshold if they are nans or greater than the threshold.
            bl_mask = np.logical_and(output_flags,np.logical_or(thisthreshold<np.abs(input_data),~np.isfinite(input_data)))
            bl_data = np.where(bl_mask,thisthreshold,input_data)
            #Calculate a rolling average array from the data with a windowsize for this iteration
            avgarray = running_mean(bl_data, window, axis=axis)
            abs_avg = np.abs(avgarray)
            #Work out the flags from the convolved data using the current threshold.
            #Flags are padded with zeros to ensure the flag array (derived from the convolved data)
            #has the same dimension as the input data.
            this_flags = abs_avg > np.expand_dims(np.take(thisthreshold,0,axis),axis)
            #Convolve the flags to be of the same width as the current window.
            convwindow = np.ones(window, dtype=np.bool)
            this_flags = np.apply_along_axis(np.convolve, axis, this_flags, convwindow)
            #"OR" the flags with the flags from the previous iteration.
            output_flags = output_flags | this_flags
        return output_flags
