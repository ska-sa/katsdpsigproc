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
    #Linearly interpolate across NaNs in y, extrapolate using numpy defaults
    #(ie. use constant). If all input data are NaNs, return 0's for all y
    nan_locs = np.isnan(y)

    if np.all(nan_locs):
        y[:] = 0.
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
        sigma = 1.4826*np.nanmedian(abs_residual[np.where(mask>0.)])
        mask=np.where(abs_residual>reject_threshold*sigma,0.0,mask)
    #Compute final background
    weight = ndimage.gaussian_filter(mask, spike_width, mode='constant', cval=0.0, truncate=3.0)
    background = ndimage.gaussian_filter(data*mask, spike_width, mode='constant', cval=0.0, truncate=3.0)/weight
    #Remove NaNs via linear interpolation
    background = np.apply_along_axis(linearly_interpolate_nans, 1, background)

    return background


def get_baseline_flags(flagger, data, flags):
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
    """Flagger that uses the SumThreshold method (Offringa, A., MNRAS, 405, 155-167, 2010)
    to detect spikes in both frequency and time axes.
    The full algorithm does the following:
        1) Average the data in the frequency dimension (axis 1) into bins of
           size self.average_freq
        2) Divide the data into overlapping sub-chunks in frequency which are
           backgrounded and thresholded independently
        3) Flag a 1d spectrum median filtered in time to get fainter contaminated
           channels.
        4) Derive a smooth 2d background through each chunk
        5) SumThreshold the background subtracted chunks in time and frequency
        6) Extend derived flags in time a frequency, via self.freq_extend and
           self.time_extend
        7) Extend flags to all times and frequencies in cases when more than
           a given fraction of samples are flagged (via self.flag_all_time_frac and
           self.flag_all_freq_frac)

    Parameters
    ----------

    outlier_nsigma: float
        Number of sigma to reject outliers when thresholding
    windows: array, int
        Size of averaging windows to use in the SumThreshold method
    background_reject: float
        Number of sigma to reject outliers when backgrounding
    background_iterations: int
        Number of iterations to use when determining a smooth background, after each
        iteration data in excess of background_reject*sigma are masked
    spike_width_time: float
        Characteristic width in dumps to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 0.
    spike_width_freq: float
        Characteristic width in channels to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 1.
    time_extend: int
        Size of window by which to extend flags in time after detection
    freq_extend: int
        Size of window by which to extend flags in frequency after detection
    freq_chunks: int
        Number of equal sized chunks to independently flag in frequnecy. Smaller
        chunks will be less affected by variations in the band in the frequency domain.
    average_freq: int
        Number of channels to average frequency before flagging. Flags will be extended
        to the frequency shape of the input data before being returned
    flag_all_time_frac: float
        Fraction of data flagged avove which to extend flags to all data in time axis.
    flag_all_freq_frac: float
        Fraction of data flagged above which to extend flags to all data in frequency axis.
    """
    def __init__(self, outlier_nsigma=4.0, windows=[1, 2, 4, 8, 16], background_reject=2.0, background_iterations=1,
                 spike_width_time=12.5, spike_width_freq=7.0, time_extend=3, freq_extend=3, freq_chunks=7,
                 average_freq=1, flag_all_time_frac=0.6, flag_all_freq_frac=0.8):
        self.outlier_nsigma = outlier_nsigma
        self.windows = windows
        self.background_reject = background_reject
        self.background_iterations = background_iterations
        self.spike_width_time = spike_width_time
        #Scale spike_width by average_freq
        self.spike_width_freq = spike_width_freq/average_freq
        self.time_extend = time_extend
        self.freq_extend = freq_extend
        self.freq_chunks = freq_chunks
        self.average_freq = average_freq
        self.flag_all_time_frac = flag_all_time_frac
        self.flag_all_freq_frac = flag_all_freq_frac
        #Falloff exponent for SumThreshold
        self.rho = 1.3

    def get_flags(self, data, flags, num_cores=8):
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

        out_flags=np.empty(data.shape, dtype=np.bool)
        async_results=[]
        p=mp.Pool(num_cores)
        for i in range(data.shape[-1]):
            async_results.append(p.apply_async(get_baseline_flags, (self, data[...,i], flags[...,i])))
        p.close()
        p.join()
        for i,result in enumerate(async_results):
            out_flags[...,i]=result.get()     

        return out_flags

    def _average_freq(self, data, flags):
        """Average the frequency axis (axis 1) of data into bins
        of size self.average_freq, ignoring flags.
        If all data in a bin is flagged in the input the data are
        averaged and the output is flagged.
        If self.average_freq does not divide the frequency axis
        of data, the last channel of avg_data will be an average of
        the remaining channels in the last bin.

        Parameters
        ----------
        data: 2D Array, real
            Input data to average.
        data: 2D Array, boolean
            Input flags of data to ignore when averaging.

        Returns
        -------
        avg_data: 2D Array, real
            The averaged data.

        avg_flags: 2D Array, boolean
            Averaged flags array- only data for which an entire averaging bin
            is flagged in the input will be flagged in avg_flags.

        """

        freq_length = data.shape[1]
        data_sum = np.add.reduceat(data * ~flags, range(0, freq_length, self.average_freq), axis=1)
        weights = np.add.reduceat(~flags, range(0, freq_length, self.average_freq), axis=1, dtype=np.float)
        avg_flags = (weights == 0.)
        #Fix weights to bin size where all data are flagged
        #This weight will be wrong for the end remainder, but data are flagged so it doesn't matter
        weights = np.where(avg_flags, float(self.average_freq), weights)
        avg_data = data_sum/weights

        return avg_data, avg_flags

    def _detect_spikes_sumthreshold(self, in_data, in_flags):
        """For a time,channel ordered input data find outliers, and extrapolate
        flags in extreme cases.

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
        #(we should have a copy of in_data,in_flags at this point)
        orig_freq_shape = in_data.shape[1]
        if self.average_freq > 1:
            in_data, in_flags = self._average_freq(in_data, in_flags)
        #Set up output flags
        out_flags = np.empty(in_data.shape, dtype=np.bool)

        #Set up frequency chunks
        freq_chunks = np.linspace(0,in_data.shape[1],self.freq_chunks+1,dtype=np.int)
        #Number of channels to pad start and end of each chunk
        #(factor of 3 is gaussian smoothing box size in getbackground)
        freq_chunk_overlap = int(self.spike_width_freq)*3

        #Loop over chunks
        for chunk_num in range(len(freq_chunks)-1):
            #Chunk the input data and flags in frequency and create output chunk
            chunk_start, chunk_end = freq_chunks[chunk_num:chunk_num+2]
            chunk_size = chunk_end - chunk_start
            if chunk_num > 0:
                chunk = slice(chunk_start-freq_chunk_overlap, chunk_end+freq_chunk_overlap)
            else:
                chunk = slice(chunk_start, chunk_end+freq_chunk_overlap)
            in_data_chunk = in_data[:,chunk]
            in_flags_chunk = np.copy(in_flags[:,chunk])
            out_flags_chunk = np.zeros_like(in_flags_chunk,dtype=np.bool)

            #Flag a 1d median frequency spectrum
            spec_flags = np.all(in_flags_chunk,axis=0).reshape((1,-1,))
            #Un-flag channels that are completely flagged to avoid nans in the median
            in_flags_chunk[:,spec_flags[0]]=False
            #Get median spectrum
            spec_data = np.nanmedian(np.where(in_flags_chunk,np.nan,in_data_chunk),axis=0).reshape((1,-1,))
            #Re-flag channels that are completely flagged in time.
            in_flags_chunk[:,spec_flags[0]]=True
            spec_background = getbackground_2d(spec_data, in_flags=spec_flags, iterations=self.background_iterations, \
                                                spike_width=(1.,self.spike_width_freq), reject_threshold=self.background_reject)
            av_dev = spec_data - spec_background
            spec_flags = self._sumthreshold(av_dev,spec_flags,1)
            #Broadcast spec_flags to timestamps
            out_flags_chunk[:] = spec_flags
            #Bootstrap the spec_flags in the mask from now on
            in_flags_chunk |= out_flags_chunk

            #Flag the data in 2d
            #Get the background in the current chunk
            background_chunk = getbackground_2d(in_data_chunk, in_flags=in_flags_chunk, iterations=self.background_iterations, \
                                                spike_width=(self.spike_width_time,self.spike_width_freq), \
                                                reject_threshold=self.background_reject)
            #Subtract background
            av_dev = in_data_chunk - background_chunk
            #SumThreshold along time axis
            time_flags_chunk = self._sumthreshold(av_dev, in_flags_chunk, 0)
            #SumThreshold along frequency axis- with time flags in the mask
            freq_flags_chunk = self._sumthreshold(av_dev, in_flags_chunk | time_flags_chunk, 1)
            #Combine all the flags
            out_flags_chunk |= time_flags_chunk | freq_flags_chunk

            #Copy out_flags from the current chunk into the correct position in out_flags (ignoring overlap)
            if chunk_num >0:
                out_flags[:,chunk_start:chunk_end] = out_flags_chunk[:,freq_chunk_overlap:freq_chunk_overlap+chunk_size]
            else:
                out_flags[:,chunk_start:chunk_end] = out_flags_chunk[:,0:chunk_size]

        #Bring flags to correct frequency shape if the input data was averaged
        #Need to chop the end to the right shape if there was a remainder after averaging
        if self.average_freq > 1:
            out_flags=np.repeat(out_flags,self.average_freq,axis=1)[:,:orig_freq_shape]

        #Extend flags in time and frequency
        if self.freq_extend > 1 or self.time_extend > 1:
            out_flags = ndimage.convolve(out_flags, np.ones((self.time_extend, self.freq_extend), dtype=np.bool), mode='reflect')

        #Flag all freqencies and times if too much is flagged.
        flag_frac_time = np.sum(out_flags, dtype=np.float, axis=0)/float(out_flags.shape[0])
        out_flags[:,np.where(flag_frac_time > self.flag_all_time_frac)[0]]=True
        flag_frac_freq = np.sum(out_flags,dtype=np.float,axis=1)/float(out_flags.shape[1])
        out_flags[np.where(flag_frac_freq > self.flag_all_freq_frac)]=True

        return out_flags


    def _sumthreshold(self, input_data, flags, axis):
        """Apply the SumThreshold method along the given axis of
        input_data. 

        Parameters
        ----------
        input_data: 2D Array, real
            Input data array (time,frequency) to flag
        flags: 2D Array, boolean
            Input flags. Used as a mask when computing the initial 
            standard deviations of the input data
        axis: int
            The axis to apply the SumThreshold operation
            0=time, 1=frequency
        window_bl: Array, int
            Window sizes to average data in each SumThreshold step
        nsigma: int
            Number of sigma to adopt for threshold.

        Returns
        -------
        output_flags: 2D Array, boolean
            The derived flags
        """

        abs_input = np.abs(input_data)
        #Get standard deviations along the axis using MAD
        estm_stdev = 1.4826 * np.nanmedian(np.abs(np.where(flags, np.nan, abs_input)),axis=axis)
        #estm_stdev will contain NaNs when all the input data are flagged
        #Set NaNs to zero to ensure these data are flagged
        estm_stdev = np.nan_to_num(estm_stdev)
        #Set up initial threshold
        threshold = self.outlier_nsigma * estm_stdev
        output_flags = np.zeros_like(flags,dtype=np.bool)
        for window in self.windows:
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
            this_flags = abs_avg >= thisthreshold_1d
            #Convolve the flags to be of the same width as the current window.
            convwindow = np.ones(window, dtype=np.bool)
            this_flags = np.apply_along_axis(np.convolve, axis, this_flags, convwindow)
            #"OR" the flags with the flags from the previous iteration.
            output_flags = output_flags | this_flags

        return output_flags
