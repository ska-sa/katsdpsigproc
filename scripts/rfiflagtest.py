#!/usr/bin/env python

"""Test script that runs RFI flagging on random or real data."""

import numpy as np
import katsdpsigproc.rfi.host
import katsdpsigproc.rfi.device
import katsdpsigproc.accel as accel
from katsdpsigproc.accel import DeviceArray
import argparse
import time
import sys

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
    parser.add_argument('--preset', '-p', choices=('small', 'medium', 'big', 'kat7'), help='(overrides other options)')
    parser.add_argument('--file', type=str, help='specify a real data file (.npy)')

    parser.add_argument_group('Parameters')
    parser.add_argument('--width', '-w', type=int, help='median filter kernel size (must be odd)', default=13)
    parser.add_argument('--sigmas', type=float, help='Threshold for detecting RFI', default=11.0)

    parser.add_argument_group('Backgrounder tuning')
    parser.add_argument('--bg-wgs', type=int, default=128, help='work group size')
    parser.add_argument('--bg-csplit', type=int, default=4, help='work items per channel')

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
            elif args.preset == 'kat7':
                args.antennas = 7
                args.channels = 8192
            elif args.preset == 'big':
                args.antennas = 64
                args.channels = 10240
            else:
                raise argparse.ArgumentError('Unexpected value for preset')
        if args.baselines is None:
            # Note: * 2 not / 2 because there are 4 polarisations
            args.baselines = args.antennas * (2 * args.antennas + 1)
        data = generate_data(args.channels, args.baselines)

    if args.width % 2 != 1:
        raise argparse.ArgumentError('Width must be odd')
    if data.shape[0] <= args.width:
        raise argparse.ArgumentError('Channels cannot be less than the filter width')

    context = None
    if not args.host:
        try:
            context = accel.create_some_context(True)
        except RuntimeError:
            print >>sys.stderr, "No devices available. Executing on the CPU."

    if context is None:
        background = katsdpsigproc.rfi.host.BackgroundMedianFilterHost(args.width)
        noise_est = katsdpsigproc.rfi.host.NoiseEstMADHost()
        threshold = katsdpsigproc.rfi.host.ThresholdSumHost(args.sigmas)
        flagger = katsdpsigproc.rfi.host.FlaggerHost(background, noise_est, threshold)
        start = time.time()
        flags = flagger(data)
        end = time.time()
        print "CPU time (ms):", (end - start) * 1000.0
    else:
        command_queue = context.create_command_queue(profile=True)
        background = katsdpsigproc.rfi.device.BackgroundMedianFilterDevice(
                command_queue, args.width, args.bg_wgs, args.bg_csplit)
        noise_est = katsdpsigproc.rfi.device.NoiseEstMADTDevice(
                command_queue, 10240)
        threshold = katsdpsigproc.rfi.device.ThresholdSumDevice(
                command_queue, args.sigmas)
        flagger = katsdpsigproc.rfi.device.FlaggerDevice(background, noise_est, threshold)

        padded_shape = flagger.min_padded_shape(data.shape)
        data_device = DeviceArray(context, data.shape, data.dtype, padded_shape)
        flags_device = DeviceArray(context, data.shape, np.uint8, padded_shape)

        data_device.set(command_queue, data)
        # Run once for warmup (allocates memory)
        flagger(data_device, flags_device)
        # Run again, timing it
        command_queue.finish()

        start_time = time.time()
        start_event = command_queue.enqueue_marker()
        flagger(data_device, flags_device)
        end_event = command_queue.enqueue_marker()
        command_queue.finish()
        end_time = time.time()
        flags = flags_device.get(command_queue)
        print "Host time (ms):  ", (end_time - start_time) * 1000.0
        try:
            device_time = end_event.time_since(start_event) * 1000.0
        except:
            # AMD CPU device doesn't seem to support profiling on marker events
            device_time = 'unknown'
        print "Device time (ms):", device_time

    print flags

if __name__ == '__main__':
    main()
