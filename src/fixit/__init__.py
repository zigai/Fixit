# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Linting framework built on LibCST, with automatic fixes."""

from .__version__ import __version__
from .api import fixit_bytes, fixit_file, fixit_paths, print_result
from .format import Formatter
from .ftypes import (
    CodePosition,
    CodeRange,
    Config,
    FileContent,
    Invalid,
    LintViolation,
    Options,
    QualifiedRule,
    Result,
    Tags,
    Valid,
)
from .rule import LintRule

__all__ = [
    "CodePosition",
    "CodeRange",
    "Config",
    "FileContent",
    "Formatter",
    "Invalid",
    "LintRule",
    "LintViolation",
    "Options",
    "QualifiedRule",
    "Result",
    "Tags",
    "Valid",
    "__version__",
    "fixit_bytes",
    "fixit_file",
    "fixit_paths",
    "print_result",
]
