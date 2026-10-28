Quadrature and metadata
=======================

NMR acquisition stores real and imaginary quadrature channels. Reader-owned
metadata describes acquisition frequencies, axes and vendor processing
profiles. The NMR unit helper can establish a ppm/frequency context:

.. code-block:: python

   from spectrochempy_nmr.units import set_nmr_context

   set_nmr_context(104.3 * scp.ur.MHz)

The context identifies the current Larmor frequency; it does not alter the
scientific meaning of the source data or replay vendor processing.
