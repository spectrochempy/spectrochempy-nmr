#!/usr/bin/env python3
"""Regenerate QF checkpoints with an external NMRPipe installation.

This is a manual evidence-generation tool. It is not imported by the plugin,
does not run in standard CI, and requires an already converted NMRPipe input.
The input must be produced by the separately documented ``bruk2pipe`` step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def run_pipeline(
    nmrpipe: Path, input_path: Path, output_path: Path, functions: list[str]
) -> None:
    """Run an explicit NMRPipe stdin/stdout pipeline."""
    environment = os.environ.copy()
    environment["PATH"] = f"{nmrpipe.parent}{os.pathsep}{environment['PATH']}"
    commands = [[str(nmrpipe), "-in", str(input_path)]]
    commands.extend([str(nmrpipe), *function.split()] for function in functions)
    commands[-1].extend(["-out", str(output_path), "-ov"])

    processes = []
    previous = None
    try:
        for command in commands:
            process = subprocess.Popen(
                command,
                env=environment,
                stdin=previous.stdout if previous is not None else None,
                stdout=subprocess.PIPE,
                stderr=None,
            )
            if previous is not None:
                previous.stdout.close()
            processes.append(process)
            previous = process
        stdout, _ = processes[-1].communicate()
        for process in processes[:-1]:
            process.wait()
        if processes[-1].returncode != 0:
            raise subprocess.CalledProcessError(
                processes[-1].returncode,
                commands[-1],
                output=stdout,
                stderr=None,
            )
        if any(process.returncode != 0 for process in processes[:-1]):
            raise subprocess.CalledProcessError(1, commands)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="NMRPipe test.fid input")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--nmrpipe", type=Path, default=Path("nmrPipe"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        args.nmrpipe,
        args.input,
        args.output / "f2.ft2",
        ["-fn ZF -zf 1 -auto", "-fn FT -nodf"],
    )
    run_pipeline(
        args.nmrpipe,
        args.input,
        args.output / "f2tp.ft2",
        ["-fn ZF -zf 1 -auto", "-fn FT -nodf", "-fn TP"],
    )
    run_pipeline(
        args.nmrpipe,
        args.output / "f2tp.ft2",
        args.output / "f1.ft2",
        ["-fn ZF -zf 1 -auto", "-fn FT -neg"],
    )
    run_pipeline(
        args.nmrpipe,
        args.output / "f1.ft2",
        args.output / "magnitude.ft2",
        ["-fn MC", "-fn TP"],
    )


if __name__ == "__main__":
    main()
