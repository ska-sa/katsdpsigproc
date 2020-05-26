#!/usr/bin/env python3

import os

import numpy as np

from katsdpsigproc.accel import Operation, IOSlot, create_some_context, build


class SumTemplate:
    def __init__(self, context):
        self.wgs = 128
        self.program = build(context, 'sum.mako', {'wgs': self.wgs},
                             extra_dirs=[os.path.dirname(__file__)])

    def instantiate(self, command_queue, size):
        return Sum(self, command_queue, size)


class Sum(Operation):
    def __init__(self, template, command_queue, size):
        if size % template.wgs:
            raise ValueError(f'size must be a multiple of {template.wgs}')
        super().__init__(command_queue)
        self.template = template
        self.slots['src'] = IOSlot((size,), np.int32)
        self.slots['dest'] = IOSlot((size // template.wgs,), np.int32)
        self.kernel = template.program.get_kernel('reduce')

    def _run(self):
        src = self.buffer('src')
        dest = self.buffer('dest')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [src.buffer, dest.buffer],
            global_size=src.shape,
            local_size=(self.template.wgs,)
        )


def main():
    ctx = create_some_context()
    queue = ctx.create_command_queue()
    op = SumTemplate(ctx).instantiate(queue, 1024)
    op.ensure_all_bound()
    src = np.random.randint(1, 100, size=op.buffer('src').shape).astype(np.int32)
    op.buffer('src').set(queue, src)
    op()
    dest = op.buffer('dest').get(queue)
    wgs = op.template.wgs
    expected = src.reshape(-1, wgs).sum(axis=1)
    np.testing.assert_equal(dest, expected)
    print(dest)


if __name__ == '__main__':
    main()
