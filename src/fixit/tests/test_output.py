# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from textwrap import dedent
from unittest import TestCase

import click
import libcst
import pytest

from fixit.ftypes import CodePosition, CodeRange, LintViolation, Result
from fixit.output import render_fixit_result


class OutputTest(TestCase):
    maxDiff = None

    def test_render_fixit_violation_with_autofix(self) -> None:
        source = dedent(
            """\
            def f():
                value = f"hello"
                return value
            """
        ).encode()
        violation = LintViolation(
            rule_name="NoRedundantFString",
            range=CodeRange(
                start=CodePosition(line=2, column=12),
                end=CodePosition(line=2, column=20),
            ),
            message="Remove redundant f-string",
            node=libcst.Name("value"),
            replacement=libcst.SimpleString('"hello"'),
        )

        rendered = render_fixit_result(
            Result(Path("example.py"), violation=violation, source=source),
            path=Path("example.py"),
        )

        assert (
            dedent(
                """\
                NoRedundantFString [*] Remove redundant f-string
                 --> example.py:2:13
                  |
                1 | def f():
                2 |     value = f"hello"
                  |             ^^^^^^^^
                3 |     return value
                  |
                help: Apply the available autofix
                """
            ).rstrip()
            == rendered
        )

    def test_render_fixit_multiline_violation(self) -> None:
        source = b"alpha beta\ngamma\ndelta\n"
        violation = LintViolation(
            rule_name="TestRule",
            range=CodeRange(
                start=CodePosition(line=1, column=6),
                end=CodePosition(line=2, column=3),
            ),
            message="Cross-line issue",
            node=libcst.Name("alpha"),
            replacement=None,
        )

        rendered = render_fixit_result(
            Result(Path("example.py"), violation=violation, source=source),
            path=Path("example.py"),
        )

        assert (
            dedent(
                """\
                TestRule Cross-line issue
                 --> example.py:1:7
                  |
                1 | alpha beta
                  |       ^^^^
                2 | gamma
                  | ^^^
                3 | delta
                  |
                """
            ).rstrip()
            == rendered
        )

    def test_render_fixit_parser_syntax_error(self) -> None:
        source = b"print)\nvalue = 1\n"
        with pytest.raises(libcst.ParserSyntaxError) as caught:
            libcst.parse_module(source.decode())

        rendered = render_fixit_result(
            Result(
                Path("broken.py"),
                violation=None,
                error=(caught.value, "traceback"),
                source=source,
            ),
            path=Path("broken.py"),
        )

        assert (
            dedent(
                """\
                invalid-syntax: tokenizer error: unmatched ')'
                 --> broken.py:1:1
                  |
                1 | print)
                  | ^^^^^
                2 | value = 1
                  |
                """
            ).rstrip()
            == rendered
        )

    def test_render_fixit_result_with_color(self) -> None:
        source = b"value = f'hello'\n"
        violation = LintViolation(
            rule_name="NoRedundantFString",
            range=CodeRange(
                start=CodePosition(line=1, column=8),
                end=CodePosition(line=1, column=16),
            ),
            message="Remove redundant f-string",
            node=libcst.Name("value"),
            replacement=libcst.SimpleString("'hello'"),
        )

        plain = render_fixit_result(
            Result(Path("example.py"), violation=violation, source=source),
            path=Path("example.py"),
        )
        colored = render_fixit_result(
            Result(Path("example.py"), violation=violation, source=source),
            path=Path("example.py"),
            color=True,
        )

        assert plain is not None
        assert colored is not None
        assert "\x1b[" in colored
        assert click.style("NoRedundantFString", fg="bright_red", bold=True) in colored
        assert f"[{click.style('*', fg='bright_cyan', bold=True)}]" in colored
        assert click.style(" --> ", fg="bright_blue", bold=True) in colored
        assert click.style("help", fg="bright_cyan", bold=True) in colored
        assert click.unstyle(colored) == plain
