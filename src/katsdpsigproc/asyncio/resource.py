"""Utilities for scheduling device operations with asyncio."""

import warnings

from ..resource import *    # noqa: F401, F403


warnings.warn("katsdpsigproc.asyncio.resource is deprecated. Use katsdpsigproc.resource instead",
              DeprecationWarning)
