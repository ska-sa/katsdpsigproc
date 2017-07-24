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

#from katsdpscripts.RTS.rfilib import plot_waterfall

#Supress warnings
#import warnings
#warnings.simplefilter('ignore')

def running_mean(x, N, axis=None):
    #Fast implementation of a running mean (array x with width N)
    #Stolen from http://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    #And modified to allow axis selection
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=axis), axis=axis)
    return np.apply_along_axis(lambda x: (x[N:] - x[:-N])/N, axis, cumsum) if axis else (cumsum[N:] - cumsum[:-N])/N

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


def getbackground_2d(data, in_flags=None, iterations=1, spike_width=(10,10,), reject_threshold=2.0):
    """Determine a smooth background over a 2d data array by
    iteratively convolving the data with elliptical Gaussians with linearly
    decreasing width from iterations*spike_width down to 1.*spike width. Outliers
    greater than reject_threshold*sigma from the background are masked on each
    iteration.
    Initial weights are set to zero at positions specified in in_flags if given.
    After the final iteration a final Gaussian smoothed background is computed
    and any stray NaNs in the background are interpolated in frequency (axis 1)
    for each timestamp (axis 0). The NaNs can appear when the the convolving
    Gaussian is completely covering masked data as the sum of convolved weights
    will be zero.

    Parameters
    ----------
    data: 2D array, float
        The input data array to be smoothed
    in_flags: 2D array, boolean (same shape as data)
        The positions in data to have zero weight in initial iteration.
    iterations: int
        The number of iterations of Gaussian smoothing
    spike_width: sequence, float
        The 1 sigma pixel widths of the smoothing gaussian (corresponding
        to the axes of data)
    reject_threshold: float
        Multiple of sigma by which to reject outliers on each iteration

    Returns
    -------
    background: 2D array, float
        The smooth background.
    """

    #Make mask array
    mask = np.ones(data.shape, dtype=np.float)

    #Mask input flags if provided
    if in_flags is not None:
        mask[in_flags]=0.0

    #Convolve with Gaussians with decreasing 1sigma width from iterations*spike_width to 1*spike_width
    for extend_factor in range(iterations, 0, -1):

        sigma = extend_factor*np.array(spike_width, dtype=np.float)

        #Get weights
        weight = ndimage.gaussian_filter(mask, sigma, mode='constant', cval=0.0, truncate=3.0)

        #Smooth background and apply weight
        background = ndimage.gaussian_filter(data*mask, sigma, mode='constant', cval=0.0, truncate=3.0)/weight

        residual = data-background
    
        #Reject outliers using MAD
        abs_residual = np.abs(residual)
        sigma = 1.4826*np.median(abs_residual[np.where(mask>0.)])
        mask=np.where(abs_residual>reject_threshold*sigma,0.0,mask)

    #Compute final background
    weight = ndimage.gaussian_filter(mask, spike_width, mode='constant', cval=0.0, truncate=3.0)
    background = ndimage.gaussian_filter(data*mask, spike_width, mode='constant', cval=0.0, truncate=3.0)/weight

    #Remove NaNs via linear interpolation
    background = np.apply_along_axis(linearly_interpolate_nans, 1, background)

    return background


def get_baseline_flags(flagger,data,flags):
    """Run flagging method for a single baseline. This is used by multiprocessing
    to avoid problems pickling an object method. It can also be used to run the flagger
    independently of multiprocessing.

    Parameters
    ----------
        flagger: A sumthreshold_flagger object             
        data: 2D array, float
            data to flag
        flags: 2D array, boolean
            prior flags to ignore 

    Returns
    -------
        a 2D array, boolean
            derived flags
    """
    return flagger._detect_spikes_sumthreshold(data,flags)


class sumthreshold_flagger():
    def __init__(self,background_iterations=1, spike_width_time=10, spike_width_freq=10, outlier_sigma_freq=4.5, outlier_sigma_time=5.5, 
                background_reject=3.0, num_windows=5, average_time=1, average_freq=1, time_extend=3, freq_extend=3, freq_chunks=7, debug=False):
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

    def get_flags(self,data,flags,num_cores=8):
        """Get flags in data array, with optional input flags of same shape
        that denote samples in data to ignore when backgrounding and deriving
        thresholds.

        Parameters
        ----------
        data: 3D array
            The input visibility data. These have their absolute value taken 
            before passing to the flagger.
        flags: 3D array, boolean
            Input flags. 
        num_cores: int
            Number of cores to use.

        Returns
        -------
        out_flags: 3D array, boolean, same shape as data
            Derived flags (True=flagged)

        """
        if self.debug: start_time=time.time()
        out_flags=np.empty(data.shape, dtype=np.bool)
        async_results=[]
        p=mp.Pool(num_cores)
        for i in range(data.shape[-1]):
            async_results.append(p.apply_async(get_baseline_flags,(self,data[...,i],flags[...,i],)))
        p.close()
        p.join()
        #for i in range(data.shape[-1]):
        #    out_flags[...,i]=get_baseline_flags(self,np.abs(data[...,i]),flags[...,i])
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
        """For a time,channel ordered input data find outliers, and extrapolate
        flags in extreme cases. The algorithm does the following:
        1) Average the data in the frequency dimension (axis 1) into bins of 
           size self.average_freq
        2) Divide the data into overlapping sub-chunks in frequency which are 
           backgrounded and thresholded independently
        3) Derive a smooth 2d background through each chunk via gaussian convolution
        4) SumThreshold the background subtracted chunks in time and frequency
        5) Extend derived flags in time a frequency, via self.freq_extend and 
           self.time_extend
        6) Extend flags to all times and frequencies in cases when more than
           a given fraction of samples are flagged (via self.flag_all_time_frac and
           self.flag_all_freq_frac)

        Parameters
        ----------
        in_data: 2D Array, float
            Array of input data (should be in (time, channel) order, and be 
            real valued- ie. complex values should have had their absolute 
            value taken before inupt.
        in_flags: 2D Array, boolean
            Array of input flags. Used to ignore points when backgrounding and 
            thresholding.

        Returns
        -------
        out_flags: 2D Array, boolean
            The output flags for the given baseline (same shape as in_data)
        """

        #Average in_data in frequency if requested
        if self.average_freq > 1:
            in_data, in_flags = self._average(in_data, in_flags)

        #Set up output flags
        out_flags = np.zeros(in_data.shape, dtype=np.bool)

        #Set up frequency chunks
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

            #Chunk the input data a flags and create output chunk
            chunk_start = freq_chunks[chunk_num]
            chunk_end = freq_chunks[chunk_num+1]
            chunk_size = chunk_end - chunk_start
            chunk = slice(chunk_start-freq_chunk_overlap,chunk_end+freq_chunk_overlap) if chunk_num > 0 \
                                                else slice(chunk_start,chunk_end+freq_chunk_overlap)
            in_data_chunk = in_data[:,chunk]
            in_flags_chunk = in_flags[:,chunk]
            out_flags_chunk = np.zeros_like(in_flags_chunk, dtype=np.bool)

            #Get the background in the current chunk
            background_chunk = getbackground_2d(in_data_chunk, in_flags=in_flags_chunk, iterations=self.background_iterations, \
                                                spike_width=(self.spike_width_time,self.spike_width_freq), \
                                                reject_threshold=self.background_reject)

            #Subtract background
            av_dev = in_data_chunk - background_chunk

            #Sumthershold along time axis
            time_flags_chunk = self._sumthreshold(av_dev, in_flags_chunk, 0, self.window_size_time, self.outlier_sigma_time)
            #Sumthreshold along frequency axis- with time flags in the mask
            freq_flags_chunk = self._sumthreshold(av_dev, in_flags_chunk | time_flags_chunk, 1, self.window_size_freq, self.outlier_sigma_freq)

            #Combine all the flags
            out_flags_chunk |= time_flags_chunk | freq_flags_chunk

            #Copy out_flags from the current chunk into the correct position in out_flags (ignoring overlap)
            out_flags[:,chunk_start:chunk_end] = out_flags_chunk[:,freq_chunk_overlap:freq_chunk_overlap+chunk_size] if chunk_num > 0 \
                                                                    else out_flags_chunk[:,0:chunk_size]

        #Bring flags to correct frequency shape if the input data was averaged
        if self.average_freq > 1:
            out_flags=np.repeat(out_flags,self.average_freq,axis=1)

        #Extend flags in time and frequency
        if self.freq_extend > 1 or self.time_extend > 1:
            out_flags = ndimage.convolve(out_flags, np.ones((self.time_extend, self.freq_extend), dtype=np.bool), mode='reflect')

        #Flag all freqencies and times if too much is flagged.
        flag_frac_time = np.sum(out_flags, dtype=np.float, axis=0)/float(out_flags.shape[0])
        out_flags[:,np.where(flag_frac_time > self.flag_all_time_frac)[0]]=True

        flag_frac_freq = np.sum(out_flags,dtype=np.float,axis=1)/float(out_flags.shape[1])
        out_flags[np.where(flag_frac_freq > self.flag_all_freq_frac)]=True

        return out_flags


    def _sumthreshold(self, input_data, flags, axis, window_bl, sigma):
        """Apply the SumThreshold method along the given axis of the
        input_data
        """

        abs_input = np.abs(input_data)

        #Get standard deviations along the axis using MAD
        estm_stdev = 1.4826 * np.nanmedian(np.abs(np.where(flags, np.nan, abs_input)),axis=axis)
        
        #estm_stdev will contain NaNs when all the input data are flagged
        #Setting NaNs to zero to ensure these data are flagged here as well
        estm_stdev = np.where(np.isnan(estm_stdev), 0.0, estm_stdev)

        #Set up initial threshold
        threshold = sigma * estm_stdev
        output_flags = np.zeros_like(flags,dtype=np.bool)
        
        for window in window_bl:

            #Stop if the window is too large 
            if window>input_data.shape[axis]: break
            
            #The threshold for this iteration is calculated from the initial threshold
            #using the equation from Offringa (2010).
            tf = pow(self.rho, np.log(window)/np.log(2.0))

            #Get the thresholds for each element of the desired axis, 
            #with an extra exis for broadcasting.
            thisthreshold_1d=np.expand_dims(threshold/tf,axis)

            #Expand thisthreshold to be same shape as input_data
            thisthreshold = thisthreshold_1d.repeat(input_data.shape[axis], axis=axis)

            #Set already flagged values to be the value of the threshold if they are outside the threshold.
            bl_mask = np.logical_or(output_flags,thisthreshold<abs_input)
            bl_data = np.where(bl_mask,thisthreshold,input_data)

            #Calculate a rolling average array from the data with the window for this iteration
            avgarray = running_mean(bl_data, window, axis=axis)
            abs_avg = np.abs(avgarray)

            #Work out the flags from the average data using the current threshold.
            this_flags = abs_avg > thisthreshold_1d

            #Convolve the flags to be of the same width as the current window.
            convwindow = np.ones(window, dtype=np.bool)
            this_flags = np.apply_along_axis(np.convolve, axis, this_flags, convwindow)

            #"OR" the flags with the flags from the previous iteration.
            output_flags = output_flags | this_flags
        
        return output_flags
