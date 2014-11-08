#!/usr/bin/env python
from katsdpsigproc import accel, transpose, copy
import numpy as np
import argparse

def benchmark(ctx, ctype, R, C, passes, copy_only):
    if ctype == 'float':
        dtype = np.float32
    elif ctype == 'float2':
        dtype = np.complex64
    elif ctype == 'float4':
        dtype = np.complex128  # Not really the same, but same size
    else:
        raise ValueError('Unhandled type')

    ctx = accel.create_some_context(True)
    if copy_only:
        template = copy.CopyTemplate(ctx, dtype, ctype)
    else:
        template = transpose.TransposeTemplate(ctx, dtype, ctype)
    queue = ctx.create_tuning_command_queue()
    proc = template.instantiate(queue, (R, C))
    proc.ensure_all_bound()
    proc()  # Warmup
    for i in range(4):
        queue.start_tuning()
        proc()
        print queue.stop_tuning()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', '-R', type=int, default=4096)
    parser.add_argument('--cols', '-C', type=int, default=4096)
    parser.add_argument('--type', '-t', type=str, choices=['float', 'float2', 'float4'], default='float2')
    parser.add_argument('--passes', type=int, default=4)
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args()

    ctx = accel.create_some_context(True)
    benchmark(ctx, args.type, args.rows, args.cols, args.passes, args.copy)

if __name__ == '__main__':
    main()
