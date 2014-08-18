#!/usr/bin/env python

"""Test script that runs RFI flagging on random or real data."""

import numpy as np
from katsdpsigproc.accel import DeviceArray
import katsdpsigproc.rfi.background
import katsdpsigproc.rfi.threshold
import katsdpsigproc.rfi.flagger
import argparse
import time

def generate_data(channels, baselines):
    real = np.random.randn(channels, baselines)
    imag = np.random.randn(channels, baselines)
    return (real + 1j * imag).astype(np.complex64)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', action='store_true', help='Compute on the host instead of with CUDA')

    parser.add_argument_group('Data selection')
    parser.add_argument('--antennas', '-a', type=int, default=7)
    parser.add_argument('--channels', '-c', type=int, default=1024)
    parser.add_argument('--baselines', '-b', type=int, help='(overrides --antennas)')
    parser.add_argument('--preset', '-p', choices=('small', 'medium', 'big'), help='(overrides other options)')
    parser.add_argument('--file', type=str, help='specify a real data file (.npy)')

    parser.add_argument_group('Parameters')
    parser.add_argument('--width', '-w', type=int, help='median filter kernel size (must be odd)', default=13)
    parser.add_argument('--sigmas', type=float, help='Threshold for detecting RFI', default=11.0)

    parser.add_argument_group('Backgrounder tuning')
    parser.add_argument('--bg-wgs', type=int, default=128, help='work group size')
    parser.add_argument('--bg-csplit', type=int, default=4, help='work items per channel')

    parser.add_argument_group('Thresholder tuning')
    parser.add_argument('--th-wgsx', type=int, default=32, help='work group size (baselines)')
    parser.add_argument('--th-wgsy', type=int, default=16, help='work group size (channels)')

    args = parser.parse_args()

    if args.file is not None:
        if args.file.endswith('.npy'):
            data = np.load(args.file)
        else:
            raise argparse.ArgumentError("Don't know how to handle " + args.file)
    else:
        if args.preset is not None:
            args.baselines = None    # Force recomputation from antennas
            if args.preset == 'small':
                args.antennas = 2
                args.channels = max(2 * args.width, 8)
            elif args.preset == 'medium':
                args.antennas = 15
                args.channels = max(2 * args.width, 128)
            elif args.preset == 'big':
                args.antennas = 64
                args.channels = 10000
            else:
                raise argparse.ArgumentError('Unexpected value for preset')
        if args.baselines is None:
            # Note: * 2 not / 2 because there are 4 polarisations
            args.baselines = args.antennas * (args.antennas + 1) * 2
        data = generate_data(args.channels, args.baselines)

    if args.width % 2 != 1:
        raise argparse.ArgumentError('Width must be odd')
    if data.shape[0] <= args.width:
        raise argparse.ArgumentError('Channels cannot be less than the filter width')

    if args.host:
        background = katsdpsigproc.rfi.background.BackgroundMedianFilterHost(args.width)
        threshold = katsdpsigproc.rfi.threshold.ThresholdMADHost(args.sigmas)
        flagger = katsdpsigproc.rfi.flagger.FlaggerHost(background, threshold)
        start = time.time()
        flags = flagger(data)
        end = time.time()
        print "CPU time (ms):", (end - start) * 1000.0
    else:
        import pycuda.autoinit
        import pycuda.driver as cuda

        ctx = pycuda.autoinit.context
        background = katsdpsigproc.rfi.background.BackgroundMedianFilterDevice(
                ctx, args.width, args.bg_wgs, args.bg_csplit)
        threshold = katsdpsigproc.rfi.threshold.ThresholdMADDevice(
                ctx, args.sigmas, args.th_wgsx, args.th_wgsy)
        flagger = katsdpsigproc.rfi.flagger.FlaggerDevice(background, threshold)

        padded_shape = flagger.min_padded_shape(data.shape)
        data_device = DeviceArray(ctx, data.shape, data.dtype, padded_shape)
        flags_device = DeviceArray(ctx, data.shape, np.uint8, padded_shape)
        start_event = cuda.Event()
        end_event = cuda.Event()

        data_device.set(data)
        # Run once for warmup (allocates memory)
        flagger(data_device, flags_device)
        # Run again, timing it
        start_event.record()
        flagger(data_device, flags_device)
        end_event.record()
        end_event.synchronize()
        print "GPU time (ms):", end_event.time_since(start_event)
        flags = flags_device.get()

    print flags

if __name__ == '__main__':
    main()
