# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest import TestCase

from click.testing import CliRunner
from pygls import uris

from fixit import __version__
from fixit.cli import main


class SmokeTest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner(mix_stderr=False)

    def test_cli_version(self) -> None:
        result = self.runner.invoke(main, ["--version"])
        expected = rf"fixit, version {__version__}"
        assert expected in result.stdout

    def test_file_with_formatting(self) -> None:
        content = dedent(
            """\
                import foo
                import bar

                def func():
                    value = f"hello world"
            """
        )
        expected_fix = dedent(
            """\
                import foo
                import bar

                def func():
                    value = "hello world"
            """
        )
        expected_format = dedent(
            """\
                import bar
                import foo


                def func():
                    value = "hello world"
            """
        )
        with TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            path = tdp / "file.py"

            with self.subTest("linting"):
                path.write_text(content)
                result = self.runner.invoke(main, ["lint", path.as_posix()], catch_exceptions=False)

                assert result.output != ""
                assert result.exit_code != 0
                assert "NoRedundantFString [*]" in result.output
                assert re.search(r" --> .*file\.py:5:13", result.output)
                assert '5 |     value = f"hello world"' in result.output
                assert "help: Apply the available autofix" in result.output
                assert content == path.read_text(), "file unexpectedly changed"

            with self.subTest("linting with diff"):
                path.write_text(content)
                result = self.runner.invoke(
                    main,
                    ["lint", "--diff", path.as_posix()],
                    catch_exceptions=False,
                )

                assert result.output != ""
                assert result.exit_code != 0
                assert "NoRedundantFString [*]" in result.output
                assert "--- a/file.py" in result.output
                assert "+++ b/file.py" in result.output

            with self.subTest("fixing"):
                path.write_text(content)
                result = self.runner.invoke(
                    main,
                    ["fix", "--automatic", path.as_posix()],
                    catch_exceptions=False,
                )

                assert result.output != ""
                assert result.exit_code == 0
                assert "NoRedundantFString [*]" in result.output
                assert re.search(r" --> .*file\.py:5:13", result.output)
                assert expected_fix == path.read_text(), "unexpected file output"

            with self.subTest("fixing with formatting"):
                (tdp / "pyproject.toml").write_text("[tool.fixit]\nformatter='ufmt'\n")

                path.write_text(content)
                result = self.runner.invoke(
                    main,
                    ["fix", "--automatic", path.as_posix()],
                    catch_exceptions=False,
                )

                assert result.output != ""
                assert result.exit_code == 0
                assert "NoRedundantFString [*]" in result.output
                assert re.search(r" --> .*file\.py:5:13", result.output)
                assert expected_format == path.read_text(), "unexpected file output"

            with self.subTest("linting via stdin"):
                result = self.runner.invoke(
                    main,
                    ["lint", "-", path.as_posix()],
                    input=content,
                    catch_exceptions=False,
                )

                assert result.output != ""
                assert result.exit_code != 0
                assert "NoRedundantFString [*]" in result.output
                assert re.search(r" --> .*file\.py:5:13", result.output)
                assert '5 |     value = f"hello world"' in result.output

            with self.subTest("fixing with formatting via stdin"):
                result = self.runner.invoke(
                    main,
                    ["fix", "-", path.as_posix()],
                    input=content,
                    catch_exceptions=False,
                )

                assert result.exit_code == 0
                assert expected_format == result.output, "unexpected stdout"

            with self.subTest("LSP"):
                path.write_text(content)
                uri = uris.from_fs_path(path.as_posix())

                initialize = (
                    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}'
                )

                did_open_template = '{{"jsonrpc":"2.0","id":1,"method":"textDocument/didOpen","params":{{"textDocument":{{"uri":{uri},"languageId":"python","version":0,"text":{content}}}}}}}'
                did_open = did_open_template.format(
                    uri=json.dumps(uri), content=json.dumps(content)
                )

                def payload(content: str) -> str:
                    return f"Content-Length: {len(content)}\r\n\r\n{content}"

                result = self.runner.invoke(
                    main,
                    ["lsp", "--debounce-interval", "0"],
                    input=payload(initialize) + payload(did_open),
                    catch_exceptions=False,
                )

                assert result.exit_code == 0
                assert re.search(
                    r"file\.py\".+\"range\".+\"start\".+\"end\".+\"severity\": 2, \"code\": \"NoRedundantFString\", \"source\": \"fixit\"",
                    result.output,
                )

    def test_this_file_is_clean(self) -> None:
        path = Path(__file__).resolve().as_posix()
        result = self.runner.invoke(main, ["lint", path], catch_exceptions=False)
        assert result.output == ""
        assert result.exit_code == 0
        assert result.stderr == "1 file clean\n"

    def test_this_project_is_clean(self) -> None:
        project_dir = Path(__file__).resolve().parent.parent.as_posix()
        result = self.runner.invoke(main, ["lint", project_dir], catch_exceptions=False)
        assert result.output == ""
        assert result.exit_code == 0

    def test_directory_with_violations(self) -> None:
        with TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            (tdp / "clean.py").write_text("name = 'Kirby'\nprint(f'hello {name}')")
            (tdp / "dirty.py").write_text("name = 'Kirby'\nprint('hello %s' % name)\n")

            result = self.runner.invoke(main, ["lint", td])
            assert "UseFstring [*] Do not use printf style formatting" in result.output
            assert re.search(r" --> .*dirty\.py:2:7", result.output)
            assert result.exit_code == 1
            assert result.stderr == "2 files checked, 1 file with errors, 1 auto-fix available\n"

    def test_directory_with_errors(self) -> None:
        with TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            (tdp / "clean.py").write_text("name = 'Kirby'\nprint(f'hello {name}')")
            (tdp / "broken.py").write_text("print)\n")

            result = self.runner.invoke(main, ["lint", td])
            assert "invalid-syntax: tokenizer error: unmatched ')'" in result.output
            assert re.search(r" --> .*broken\.py:1:1", result.output)
            assert result.exit_code == 2

    def test_directory_with_violations_and_errors(self) -> None:
        with TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            (tdp / "clean.py").write_text("name = 'Kirby'\nprint(f'hello {name}')")
            (tdp / "dirty.py").write_text("name = 'Kirby'\nprint('hello %s' % name)\n")
            (tdp / "broken.py").write_text("print)\n")

            result = self.runner.invoke(main, ["lint", td])
            assert "UseFstring [*] Do not use printf style formatting" in result.output
            assert re.search(r" --> .*dirty\.py:2:7", result.output)
            assert "invalid-syntax: tokenizer error: unmatched ')'" in result.output
            assert re.search(r" --> .*broken\.py:1:1", result.output)
            assert result.exit_code == 3

    def test_directory_with_autofixes(self) -> None:
        with TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            clean = tdp / "clean.py"
            clean.write_text(
                dedent(
                    """
                    GLOBAL = 'hello'

                    def foo():
                        value = 'test'
                        if value is False:
                            pass
                    """
                )
            )
            single = tdp / "single.py"
            single.write_text(
                dedent(
                    """
                    GLOBAL = f'hello'

                    def foo():
                        value = 'test'
                        if value is False:
                            pass
                    """
                )
            )
            multi = tdp / "multi.py"
            multi.write_text(
                dedent(
                    """
                    GLOBAL = f'hello'

                    def foo():
                        value = f'test'
                        if value == False:
                            pass
                    """
                )
            )

            expected = clean.read_text()

            result = self.runner.invoke(main, ["fix", "--automatic", td])
            errors = defaultdict(list)
            pattern = re.compile(
                r"(?m)^(?P<header>[^\n]+)\n --> (?P<path>.+):(?P<line>\d+):(?P<col>\d+)$"
            )
            for match in pattern.finditer(result.output):
                rule_name = match.group("header").split(" ", 1)[0]
                location = f"{match.group('line')}:{match.group('col')} {rule_name}"
                errors[Path(match.group("path"))].append(location)

            with self.subTest("clean"):
                assert [] == errors[clean]
                assert expected == clean.read_text()

            with self.subTest("single fix"):
                assert [
                    "2:10 NoRedundantFString",
                ] == sorted(errors[single])
                assert expected == single.read_text()

            with self.subTest("multiple fixes"):
                assert [
                    "2:10 NoRedundantFString",
                    "5:13 NoRedundantFString",
                    "6:8 CompareSingletonPrimitivesByIs",
                ] == sorted(errors[multi])
                assert expected == multi.read_text()

    def test_lint_directory_with_no_rules_enabled(self) -> None:
        content = dedent(
            """\
                import foo
                import bar

                def func():
                    value = f"hello world"
            """
        )
        with self.subTest("lint"), TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            path = tdp / "file.py"

            (tdp / "pyproject.toml").write_text("[tool.fixit]\ndisable=['fixit.rules']\n")

            path.write_text(content)
            result = self.runner.invoke(
                main,
                ["lint", path.as_posix()],
                catch_exceptions=False,
            )

            assert result.output == ""
            assert result.exit_code == 0

        with self.subTest("fix"), TemporaryDirectory() as td:
            tdp = Path(td).resolve()
            path = tdp / "file.py"

            (tdp / "pyproject.toml").write_text("[tool.fixit]\ndisable=['fixit.rules']\n")

            path.write_text(content)
            result = self.runner.invoke(
                main,
                ["fix", "--automatic", path.as_posix()],
                catch_exceptions=False,
            )

            assert result.output == ""
            assert result.exit_code == 0
