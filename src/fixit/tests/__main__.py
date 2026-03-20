# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([Path(__file__).parent.as_posix(), *sys.argv[1:]]))
