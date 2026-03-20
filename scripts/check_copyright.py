# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections.abc import Iterable
from pathlib import Path
from shutil import which
from subprocess import run

# Use the copyright header from this file as the benchmark for all files
EXPECTED_HEADER = "\n".join(line for line in Path(__file__).read_text().splitlines()[:4])


def tracked_files() -> Iterable[Path]:
    git_executable = which("git")
    if git_executable is None:
        raise RuntimeError("git executable not found")

    proc = run(  # noqa: S603 - fixed git command
        [git_executable, "ls-tree", "-r", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    yield from (
        path
        for line in proc.stdout.splitlines()
        if (path := Path(line)) and path.is_file() and path.suffix in (".py", ".sh")
    )


def main() -> None:
    error = False
    for path in tracked_files():
        content = path.read_text("utf-8")
        if EXPECTED_HEADER not in content:
            print(f"Missing or incomplete copyright in {path}")
            error = True
    sys.exit(1 if error else 0)


if __name__ == "__main__":
    main()
