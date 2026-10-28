Hypercomplex data in multidimensional NMR
=========================================

The independent ``spectrochempy-hypercomplex`` plugin represents two complex
dimensions as quaternion data. Its four real components are:

* ``RR`` — real/real;
* ``RI`` — real/imaginary;
* ``IR`` — imaginary/real;
* ``II`` — imaginary/imaginary.

Use the dynamic ``dataset.hyper`` accessor without coupling the package to the
NMR plugin:

.. code-block:: python

   if not dataset.hyper.is_quaternion:
       dataset.hyper.set_quaternion(inplace=True)
   rr = dataset.hyper.component("RR")
   ri = dataset.hyper.RI

This representation is supported for relevant 1D/2D reading paths. It does
not turn ``Experiment.process()`` or a single core ``fft()`` call into a
validated general 2D processing pipeline.
