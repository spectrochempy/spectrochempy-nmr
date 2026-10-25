# Repository extraction provenance

This repository is a candidate for the independent maintenance of the existing
`spectrochempy-nmr` distribution. Package publication still belongs to the
SpectroChemPy monorepo until the release handover is explicitly completed.

The repository name uses an underscore (`spectrochempy_nmr`) for consistency
with the organization's other workspaces. The distribution name remains
`spectrochempy-nmr`; the import package is `spectrochempy_nmr`.

## Source mapping

- Source repository: https://github.com/spectrochempy/spectrochempy
- Source commit: `cdb087a24f06b8c83ea91ea036192304db9329e3`
- Source directory: `plugins/spectrochempy-nmr`
- Extracted history tip: `773f237f40a07a688d7cdbb4d7d01c871d2c7e9a`
- Identical source/extracted tree: `cbe89a9b43cfc35bee31c0d7281aa6a252acb979`

History was extracted with `git subtree split --prefix=plugins/spectrochempy-nmr`
at the source commit. Authors and relevant directory history are retained;
commit IDs change because paths and parent histories change. This history covers
the plugin directory, not all earlier implementations elsewhere in the monorepo.

The initial repository setup changes documentation and CI only. Runtime sources,
tests, examples, package version and vendored parsers match the source baseline.

## Validation and remaining handover

The pre-extraction detached Python 3.13 wheel baseline has 400 passing tests and
94 warnings with the maintainer's external dataset corpus. The public CI builds
sdist/wheel artifacts once, verifies both notices and runtime file parity, then
runs 70 synthetic tests on Python 3.11–3.13 against released core 1.1.0 (declared
floor) and 1.1.3, plus one integration leg on Python 3.13 against the pinned core
baseline and its HyperComplex plugin. It is not a substitute for the complete
real-data baseline. Dataset permissions and a pinned CI fixture manifest still
require resolution.

Required status checks protect `master`; they cover artifact construction, the
released-core compatibility matrix and the pinned-core integration leg.

No PyPI/Conda publishing workflow is installed. Before publishing a new version,
disable the monorepo's NMR publishing routes and old Trusted Publisher, then
configure the new repository's release workflow and publisher. Retain a single
publication authority. Version 0.1.15 is historical and must not be republished.

Documentation remains intended for integration into the main SpectroChemPy site
through a pinned export. That cross-repository integration is not yet wired.
Migration to nmrglue-ng and indirect Agilent TPPI convention changes remain
separate work.
