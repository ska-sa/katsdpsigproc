#!/usr/bin/env python
import katsdpsigproc.accel

ctx = katsdpsigproc.accel.create_some_context(interactive=True)
print(f'Successfully created context on {ctx.device.name} ({ctx.device.platform_name})')
