# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from click.testing import CliRunner

from fixit.cli import main


class CliTest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner(mix_stderr=False)

    def test_upgrade_command_removed(self) -> None:
        result = self.runner.invoke(main, ["upgrade"], catch_exceptions=False)
        assert result.exit_code == 2
        assert "No such command 'upgrade'" in result.stderr
