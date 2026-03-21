# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path

from fixit.ftypes import LSPOptions, Options, QualifiedRule
from fixit.lsp import LSP


def test_lsp_config_cache_invalidation(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    target_path = tmp_path / "sample.py"
    target_path.write_text("x = 1\n")

    config_path.write_text(
        """
[tool.fixit]
root = true
disable = ["fixit.rules"]
"""
    )

    lsp = LSP(Options(), LSPOptions(tcp=None, ws=None, stdio=False, debounce_interval=0))

    first = lsp.load_config(target_path)
    assert QualifiedRule("fixit.rules") in first.disable

    time.sleep(0.01)
    config_path.write_text(
        """
[tool.fixit]
root = true
"""
    )

    second = lsp.load_config(target_path)
    assert first is not second
    assert second.disable == []
