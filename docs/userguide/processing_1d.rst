Validated 1D processing
=======================

Wrap a dataset in :class:`~spectrochempy_nmr.Experiment` and select explicit
processing parameters:

.. code-block:: python

   experiment = scp.nmr.Experiment(fid)
   result = experiment.process(apodization="em", lb=5.0, phase=None)

The supported apodization modes are ``em``, ``gm`` and ``sp``. The result
records the SpectroChemPy-owned call in
``result.meta.nmr_processing["scp_processing"]``. ``requested`` contains
explicit user arguments and ``applied`` records operations actually executed.

``Experiment.process()`` is validated for 1D experiments only and raises for
general multidimensional datasets. A pseudo-2D series is a collection of
ordinary 1D spectra and may be processed one spectrum at a time; it is not a
general 2D Fourier-processing chain.

Generic operations such as :external+core:py:func:`spectrochempy.fft`,
apodization methods, :external+core:py:class:`spectrochempy.NDDataset`,
:external+core:py:class:`spectrochempy.Coord`,
:external+core:py:class:`spectrochempy.CoordSet`, and units belong to the
core documentation.
This site documents their NMR use without duplicating their technical API.
