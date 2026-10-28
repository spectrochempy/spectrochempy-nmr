#!/usr/bin/env python3
"""Generate the synthetic States-TPPI NMRPipe oracle.

This manual tool requires NMRPipe 13 and the external nmrglue-ng reader used
for evidence generation. It is not imported by the plugin and is not run in
CI. The committed NPZ contains only deterministic synthetic data and NMRPipe
checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import nmrglue as ng


N_F1 = 32
N_F2 = 256
N_F2_FT = 512
N_F1_FT = 64


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _run_pipeline(
    nmrpipe: Path,
    input_path: Path,
    output_path: Path,
    functions: list[list[str]],
    *,
    environment: dict[str, str],
) -> None:
    commands = [[str(nmrpipe), "-in", str(input_path)]]
    commands.extend([str(nmrpipe), *function] for function in functions)
    commands[-1].extend(["-out", str(output_path), "-ov"])
    processes = []
    previous = None
    try:
        for command in commands:
            process = subprocess.Popen(
                command,
                cwd=input_path.parent,
                env=environment,
                stdin=previous.stdout if previous is not None else None,
                stdout=subprocess.PIPE,
            )
            if previous is not None:
                previous.stdout.close()
            processes.append(process)
            previous = process
        processes[-1].communicate()
        for process in processes[:-1]:
            process.wait()
        if any(process.returncode for process in processes):
            raise subprocess.CalledProcessError(1, commands)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()


def _template_table(path: Path) -> None:
    path.write_text(
        """VARS INDEX X_AXIS Y_AXIS DX DY X_PPM Y_PPM X_HZ Y_HZ XW YW XW_HZ YW_HZ X1 X3 Y1 Y3 HEIGHT DHEIGHT VOL PCHI2 TYPE ASS CLUSTID MEMCNT
FORMAT %5d %9.3f %9.3f %6.3f %6.3f %8.3f %8.3f %7.3f %7.3f %7.3f %7.3f %8.3f %8.3f %4d %4d %4d %4d %+e %+e %+e %.5f %d %s %4d %4d
    1 5 50 0 0 5 50 0 0 1 1 1 1 1 1 1 1 1 0 1 0 1 None 1 1
""",
        encoding="utf-8",
    )


def _signal() -> tuple[np.ndarray, dict[str, object]]:
    t1 = np.arange(N_F1)[:, None]
    t2 = np.arange(N_F2)[None, :]
    first = np.exp(-t1 / 45 - t2 / 180) * np.exp(
        2j * np.pi * (7.25 * t1 / N_F1 + 61.5 * t2 / N_F2) + 0.31j
    )
    first += (
        0.63
        * np.exp(-t1 / 27 - t2 / 115)
        * np.exp(2j * np.pi * (-11.5 * t1 / N_F1 + 137.25 * t2 / N_F2) - 0.47j)
    )
    second = (
        0.41
        * np.exp(-t1 / 38 - t2 / 150)
        * np.exp(2j * np.pi * (13.5 * t1 / N_F1 - 83.75 * t2 / N_F2) + 0.83j)
    )
    second += 0.22 * np.conjugate(first)
    components = np.stack(
        [first.real, first.imag, second.real, second.imag], axis=-1
    ).astype(np.float32)
    parameters = {
        "f1_points": [7.25, -11.5, 13.5],
        "f2_points": [61.5, 137.25, -83.75],
        "amplitudes": [1.0, 0.63, 0.41, 0.22],
        "phases_rad": [0.31, -0.47, 0.83, 0.0],
        "damping_points": [[45, 180], [27, 115], [38, 150]],
        "noise": "none",
    }
    return components, parameters


def _canonical_components(final: np.ndarray) -> np.ndarray:
    """Decode NMRPipe's final interleaving into RR, RI, IR, II."""
    first = np.conjugate(final[0::2]).T
    second = np.conjugate(final[1::2]).T
    first = _reverse_roll(first)
    second = _reverse_roll(second)
    return np.stack([first.real, first.imag, second.real, second.imag], axis=-1).astype(
        np.float32
    )


def _reverse_roll(data: np.ndarray) -> np.ndarray:
    return np.roll(data[..., ::-1], 1, axis=-1)


def _header_profile(header: dict[str, float]) -> dict[str, float | int]:
    keys = (
        "FDDIMCOUNT",
        "FDTRANSPOSED",
        "FDSIZE",
        "FDSPECNUM",
        "FDF1TDSIZE",
        "FDF2TDSIZE",
        "FDF1FTSIZE",
        "FDF2FTSIZE",
        "FDF1QUADFLAG",
        "FDF2QUADFLAG",
        "FDF1AQSIGN",
        "FDF2AQSIGN",
    )
    return {key: header[key] for key in keys}


def generate(output: Path, nmrpipe: Path) -> None:
    environment = os.environ.copy()
    environment["PATH"] = f"{nmrpipe.parent}{os.pathsep}{environment['PATH']}"
    with tempfile.TemporaryDirectory(prefix="states-tppi-oracle-") as directory:
        work = Path(directory)
        table = work / "template.tab"
        _template_table(table)
        _run(
            [
                "simTimeND",
                "-ndim",
                "2",
                "-in",
                str(table),
                "-noise",
                "0",
                "-scale",
                "1",
                "-notdd",
                "-ts",
                "-aq2D",
                "States",
                "-xN",
                "512",
                "-yN",
                "64",
                "-xT",
                "256",
                "-yT",
                "32",
                "-xMODE",
                "Complex",
                "-yMODE",
                "States-TPPI",
                "-xSW",
                "10000",
                "-ySW",
                "2000",
                "-xOBS",
                "600",
                "-yOBS",
                "60",
                "-xCAR",
                "4.8",
                "-yCAR",
                "118",
                "-xLAB",
                "HN",
                "-yLAB",
                "N15",
                "-out",
                str(work / "template.fid"),
                "-ov",
            ],
            cwd=work,
            environment=environment,
        )
        header, _ = ng.pipe.read(str(work / "template.fid"))
        components, signal_parameters = _signal()
        raw = np.empty((2 * N_F1, N_F2), dtype=np.complex64)
        raw[0::2] = components[..., 0] + 1j * components[..., 1]
        raw[1::2] = components[..., 2] + 1j * components[..., 3]
        ng.pipe.write(str(work / "input.fid"), header, raw, overwrite=True)
        _run_pipeline(
            nmrpipe,
            work / "input.fid",
            work / "f2.ft2",
            [["-fn", "ZF", "-zf", "1", "-auto"], ["-fn", "FT", "-nodf"]],
            environment=environment,
        )
        _run_pipeline(
            nmrpipe,
            work / "f2.ft2",
            work / "tp.ft2",
            [["-fn", "TP"]],
            environment=environment,
        )
        _run_pipeline(
            nmrpipe,
            work / "tp.ft2",
            work / "final.ft2",
            [["-fn", "ZF", "-zf", "1", "-auto"], ["-fn", "FT", "-alt"]],
            environment=environment,
        )
        input_header, input_data = ng.pipe.read(str(work / "input.fid"))
        f2_header, f2_data = ng.pipe.read(str(work / "f2.ft2"))
        tp_header, tp_data = ng.pipe.read(str(work / "tp.ft2"))
        final_header, final_data = ng.pipe.read(str(work / "final.ft2"))
        arrays = {
            "input": input_data,
            "f2": f2_data,
            "tp": tp_data,
            "final": final_data,
            "input_components": components,
            "f2_components": np.stack(
                [
                    _reverse_roll(f2_data[0::2].real),
                    _reverse_roll(f2_data[0::2].imag),
                    _reverse_roll(f2_data[1::2].real),
                    _reverse_roll(f2_data[1::2].imag),
                ],
                axis=-1,
            ).astype(np.float32),
            "tp_cross": np.stack(
                [
                    _reverse_roll(tp_data[0::2].T),
                    _reverse_roll(tp_data[1::2].T),
                ],
                axis=-1,
            ).astype(np.complex64),
            "final_components": _canonical_components(final_data),
        }
        output.mkdir(parents=True, exist_ok=True)
        fixture = output / "states_tppi_oracle.npz"
        np.savez_compressed(fixture, **arrays)
        manifest = {
            "nmrpipe": "13.0 Rev 2026.072.12.03 64-bit",
            "binary": "nmrPipe/simTimeND (manual generation only)",
            "encoding": {"yMODE": "States-TPPI", "yALT": 2, "aq2D": "States"},
            "trace_order": "trace[2a]=RR+i*RI; trace[2a+1]=IR+i*II",
            "pipeline": [
                "simTimeND synthetic header profile",
                "ZF -zf 1 -auto | FT -nodf",
                "TP",
                "ZF -zf 1 -auto | FT -alt",
            ],
            "signal": signal_parameters,
            "conventions": {
                "nmrpipe_final_storage": "[q0, q2, q1, q3] interleaving",
                "decode": "conjugate final even/odd, transpose, reverse+roll F2, then [real, imag, real, imag]",
                "plugin_order": ["RR", "RI", "IR", "II"],
            },
            "headers": {
                "input": _header_profile(input_header),
                "f2": _header_profile(f2_header),
                "tp": _header_profile(tp_header),
                "final": _header_profile(final_header),
            },
            "files": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "redistribution": "deterministic synthetic signal and NMRPipe checkpoints only",
        }
        (output / "states_tppi_oracle_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--nmrpipe", type=Path, default=Path("nmrPipe"))
    args = parser.parse_args()
    generate(args.output, args.nmrpipe)


if __name__ == "__main__":
    main()
