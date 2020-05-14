Built-in operations
===================

There are a small number of built-in :doc:`operations` provided. Some of them
are quite general-purpose, which some others were developed for a specific
radio-astronomy pipeline and might not be useful elsewhere, but could be seen
as examples.

At present there is no step-by-step user guide for these operations, so you
will need to refer to the linked API docs to learn how to use them. The links
below are to the operation template classes, which contain the high-level
descriptions of the operations; the corresponding operation classes document
the slots they provide.

- :class:`.FillTemplate`: fill a buffer with a constant value.

- :class:`.TransposeTemplate`: transpose a 2D array.

- :class:`.HReduceTemplate`: apply a reduction (such as a sum) to each row of a
  2D array.

- :class:`.MaskedSumTemplate`: compute a weighted sum of each column of a 2D
  array.

- :class:`.Percentile5Template`: compute percentiles of the data in each row of
  a 2D array.

RFI flagging
------------
The :mod:`katsdpsigproc.rfi.device` module contains a number of classes for
detecting RFI (Radio Frequency Interference) in radio-astronomy data. There are
also CPU-only equivalents of each of these classes (mostly for testing rather
than high performance) in :mod:`katsdpsigproc.rfi.host`. There is also a more
sophisticated and optimized flagger in :mod:`katsdpsigproc.rfi.twodflag`
(so-called because it looks for anomalies in both time and frequency).
