#!/usr/bin/env python

################################################################################
# Copyright (c) 2014-2020, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Test script that runs RFI flagging on random or real data."""

import argparse
import time
import sys
import json
import concurrent.futures

import numpy as np

import katsdpsigproc.rfi.host
import katsdpsigproc.rfi.device
import katsdpsigproc.rfi.twodflag
import katsdpsigproc.accel as accel


def generate_data(times, channels, baselines):
    rs = np.random.RandomState(seed=1)
    shape = (channels, baselines) if times is None else (times, channels, baselines)
    # This is done a row at a time to keep memory usage low
    out = np.empty(shape, np.complex64)
    for i in range(shape[0]):
        real = rs.standard_normal(size=shape[1:]).astype(np.float32)
        imag = rs.standard_normal(size=shape[1:]).astype(np.float32)
        out[i] = real + 1j * imag
    return out


def benchmark1d(args, data):
    if args.width % 2 != 1:
        raise argparse.ArgumentError('Width must be odd')
    if data.shape[0] <= args.width:
        raise argparse.ArgumentError('Channels cannot be less than the filter width')

    context = None
    if not args.host:
        try:
            context = accel.create_some_context(True)
        except RuntimeError:
            print("No devices available. Executing on the CPU.", file=sys.stderr)

    if context is None:
        background = katsdpsigproc.rfi.host.BackgroundMedianFilterHost(args.width)
        noise_est = katsdpsigproc.rfi.host.NoiseEstMADHost()
        threshold = katsdpsigproc.rfi.host.ThresholdSumHost(args.sigmas)
        flagger = katsdpsigproc.rfi.host.FlaggerHost(background, noise_est, threshold)
        start = time.time()
        flags = flagger(data)
        end = time.time()
        print("CPU time (ms):", (end - start) * 1000.0)
    else:
        command_queue = context.create_command_queue(profile=True)
        background = katsdpsigproc.rfi.device.BackgroundMedianFilterDeviceTemplate(
            context, args.width)
        noise_est = katsdpsigproc.rfi.device.NoiseEstMADTDeviceTemplate(
            context, 10240)
        threshold = katsdpsigproc.rfi.device.ThresholdSumDeviceTemplate(context)
        template = katsdpsigproc.rfi.device.FlaggerDeviceTemplate(background, noise_est, threshold)
        flagger = template.instantiate(command_queue, data.shape[0], data.shape[1],
                                       threshold_args={'n_sigma': args.sigmas})
        flagger.ensure_all_bound()

        data_device = flagger.buffer('vis')
        flags_device = flagger.buffer('flags')

        data_device.set(command_queue, data)
        # Run once for warmup (allocates memory)
        flagger()
        # Run again, timing it
        command_queue.finish()

        start_time = time.time()
        start_event = command_queue.enqueue_marker()
        flagger()
        end_event = command_queue.enqueue_marker()
        command_queue.finish()
        end_time = time.time()
        flags = flags_device.get(command_queue)
        print("Host time (ms):  ", (end_time - start_time) * 1000.0)
        try:
            device_time = end_event.time_since(start_event) * 1000.0
        except Exception:
            # AMD CPU device doesn't seem to support profiling on marker events
            device_time = 'unknown'
        print("Device time (ms):", device_time)
    return flags


def benchmark2d(args, data):
    flagger = katsdpsigproc.rfi.twodflag.SumThresholdFlagger(**args.params)
    in_flags = np.zeros(data.shape, np.bool_)
    if args.pool == 'none':
        pool = None
    elif args.pool == 'process':
        pool = concurrent.futures.ProcessPoolExecutor(args.workers)
    elif args.pool == 'thread':
        pool = concurrent.futures.ThreadPoolExecutor(args.workers)
    else:
        raise argparse.ArgumentError(f'unhandled value {args.pool} for --pool')
    # Warmup
    try:
        flagger.get_flags(data[:2], in_flags[:2], pool=pool)
        start = time.time()
        flags = flagger.get_flags(data, in_flags, pool=pool)
        end = time.time()
        print("CPU time (ms):", (end - start) * 1000.0)
    finally:
        if pool is not None:
            pool.shutdown()
    return flags


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', action='store_true',
                        help='Compute on the host instead of with CUDA/OpenCL')

    parser.add_argument_group('Data selection')
    parser.add_argument('--antennas', '-a', type=int, default=7)
    parser.add_argument('--channels', '-c', type=int, default=1024)
    parser.add_argument('--baselines', '-b', type=int, help='(overrides --antennas)')
    parser.add_argument('--times', '-t', type=int, help='time samples (uses 2D flagger)')
    parser.add_argument('--preset', '-p', choices=('small', 'medium', 'big', 'kat7'),
                        help='(overrides other options)')
    parser.add_argument('--file', type=str, help='specify a real data file (.npy)')

    parser.add_argument_group('Parameters (1D flagger)')
    parser.add_argument('--width', '-w', type=int, help='median filter kernel size (must be odd)',
                        default=13)
    parser.add_argument('--sigmas', type=float, help='threshold for detecting RFI', default=11.0)

    parser.add_argument_group('Parameters (2D flagger)')
    parser.add_argument('--params', type=json.loads, default={}, help='JSON dict of parameters')
    parser.add_argument('--pool', choices=('none', 'process', 'thread'),
                        help='parallelization method')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')

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
            args.baselines = args.antennas * (args.antennas + 1) * 2
        data = generate_data(args.times, args.channels, args.baselines)

    if args.times is None:
        flags = benchmark1d(args, data)
    else:
        flags = benchmark2d(args, data)
    print('{:.4f}% flagged'.format(100.0 * np.sum(flags) / flags.size))


if __name__ == '__main__':
    main()
