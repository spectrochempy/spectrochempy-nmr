# NMRPipe QF oracle regeneration

`regenerate_qf_oracle.py` is an external-software helper for regenerating the
small QF checkpoints used by the NMRPipe validation campaign. It is not part
of the plugin runtime and is not executed by standard CI.

The input must be the **post-`EXT` synthetic QF input** (`input.fid`) described
by the committed oracle manifest. The helper does not generate the synthetic
signal, construct a `bruk2pipe` header, or perform `EXT`; those are controlled
provenance steps used when producing the committed fixture. It writes four
NMRPipe checkpoints:

```text
f2.ft2          direct-dimension zero-fill + FT -nodf
f2tp.ft2        explicit transposition
f1.ft2          indirect zero-fill + FT -neg
magnitude.ft2   MC magnitude + final transposition
```

Example:

```bash
python tools/nmrpipe/regenerate_qf_oracle.py \
    /path/to/synthetic-input.fid /tmp/nmrpipe-qf-oracle \
    --nmrpipe /home/christian/pipe/nmrbin.linux239_64/nmrPipe
```

The generated files remain external evidence. Do not commit experimental
corpora or generated binary outputs without a separate provenance and
redistribution decision. The committed synthetic fixture is packaged
separately as `tests/data/nmrpipe/qf_oracle.npz`; this helper intentionally
does not overwrite that package or its manifest.
