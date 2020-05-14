import numpy as np
import katsdpsigproc.accel
import katsdpsigproc.fill
import katsdpsigproc.reduce


class FillReduceTemplate:
    def __init__(self, context):
        self.fill = katsdpsigproc.fill.FillTemplate(context, np.float32, 'float')
        self.hreduce = katsdpsigproc.reduce.HReduceTemplate(context, np.float32, 'float', 'a+b', '0.0f')

    def instantiate(self, queue, shape):
        return FillReduce(self, queue, shape)


class FillReduce(katsdpsigproc.accel.OperationSequence):
    def __init__(self, template, queue, shape):
        self.fill = template.fill.instantiate(queue, shape)
        self.hreduce = template.hreduce.instantiate(queue, shape)
        operations = [
            ('fill', self.fill),
            ('hreduce', self.hreduce)
        ]
        compounds = {
            'src': ['fill:data', 'hreduce:src'],
            'dest': ['hreduce:dest']
        }
        super().__init__(queue, operations, compounds)
        self.template = template

    def __call__(self, fill_value):
        self.fill.set_value(fill_value)
        super().__call__()


ctx = katsdpsigproc.accel.create_some_context()
queue = ctx.create_command_queue()
op_template = FillReduceTemplate(ctx)
op = op_template.instantiate(queue, (10, 5))
op(42)
print(op.buffer('dest').get(queue))
