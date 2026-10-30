# Phase 2b validation record

This record describes the clean local RTD-like validation performed for the
Phase 2b working tree. It is not a substitute for an actual Read the Docs
build; the RTD project is not modified by this phase.

The validated functional documentation state is commit
`b10ab3c9058cac3c4669d858b3015e421a1c13a8`. Commit `17c886f` records only the
source/manifest provenance for that validated state; it does not alter the
theme, gallery, fixture fetcher or build behavior.

## Environment

```text
Python 3.13.13
SpectroChemPy core 1.1.3
spectrochempy-nmr 0.2.0 from checkout
spectrochempy-hypercomplex 0.2.0
Sphinx 7.4.7
pydata-sphinx-theme 0.17.1
sphinx-gallery 0.19.0
numpydoc 1.8.0
```

## Commands

```console
python -m venv /tmp/opencode/nmr-theme-gallery-venv
python -m pip install -r docs/requirements-rtd.txt
python -m pip install .
python -m pip check
python docs/fetch_gallery_data.py
READTHEDOCS=True MPLBACKEND=Agg \
  python -m sphinx -W --keep-going -b html docs docs/_build-rtd
```

The build was run after deleting `docs/examples/gallery`,
`docs/sg_execution_times.rst` and the output directory, so Sphinx-Gallery could
not reuse generated source or execution caches.

## Results

- `pip check`: **PASS**;
- Sphinx strict build: **PASS**;
- Sphinx-Gallery execution: **6/6**;
- generated figures: **24**;
- generated Python output blocks: present in the gallery pages;
- CP/2D pages: **0**;
- `objects.inv`: **PASS**;
- elapsed build time: **17.53 s**;
- peak RSS: **265536 KiB** (approximately 259 MiB).

The local non-executing mode was also rebuilt from a clean generated-source
tree with `READTHEDOCS` and `NMR_DOCS_EXECUTE` unset; it passed strictly and
reported that no examples were executed.
