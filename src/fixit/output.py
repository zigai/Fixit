from __future__ import annotations

import re
import sys
from pathlib import Path

import click
from libcst import ParserSyntaxError

from .ftypes import CodePosition, CodeRange, FileContent, LintViolation, Result

_PARSER_ERROR_PREFIX = re.compile(r"^parser error:\s*", re.IGNORECASE)
_PARSER_LOCATION_PREFIX = re.compile(r"^error at \d+:\d+:\s*", re.IGNORECASE)


def render_fixit_result(result: Result, *, path: Path, color: bool = False) -> str | None:
    """Render a result using the default terminal presentation."""
    if result.violation:
        return _render_violation(path, result.violation, result.source, color=color)

    if result.error:
        error, _ = result.error
        if isinstance(error, ParserSyntaxError):
            return _render_syntax_error(path, error, result.source, color=color)

    return None


def _render_violation(
    path: Path, violation: LintViolation, source: FileContent | None, *, color: bool
) -> str:
    header = _error_style(violation.rule_name, color=color)
    if violation.autofixable:
        header += f" {_fix_marker(color=color)}"
    header += f" {violation.message}"
    lines = _render_block(
        header=header,
        path=path,
        code_range=violation.range,
        source=source,
        help_message=("Apply the available autofix" if violation.autofixable else None),
        color=color,
    )
    return "\n".join(lines)


def _render_syntax_error(
    path: Path, error: ParserSyntaxError, source: FileContent | None, *, color: bool
) -> str:
    source_lines = _decode_source_lines(source)
    line_text = source_lines[error.raw_line - 1] if error.raw_line - 1 < len(source_lines) else ""
    end_column = _find_syntax_error_end_column(line_text, error.raw_column)
    error_range = CodeRange(
        start=CodePosition(line=error.raw_line, column=error.raw_column),
        end=CodePosition(line=error.raw_line, column=end_column),
    )
    lines = _render_block(
        header=(
            f"{_error_style('invalid-syntax', color=color)}: "
            f"{_normalize_parser_message(error.message)}"
        ),
        path=path,
        code_range=error_range,
        source=source,
        color=color,
    )
    return "\n".join(lines)


def _render_block(
    *,
    header: str,
    path: Path,
    code_range: CodeRange,
    source: FileContent | None,
    help_message: str | None = None,
    color: bool,
) -> list[str]:
    source_lines = _decode_source_lines(source)
    display_column = code_range.start.column + 1
    lines = [
        header,
        (
            f"{_line_no_style(' --> ', color=color)}"
            f"{path.as_posix()}:{code_range.start.line}:{display_column}"
        ),
    ]

    if not source_lines:
        if help_message:
            lines.extend(
                [
                    _line_no_style("  |", color=color),
                    f"{_help_style('help', color=color)}: {help_message}",
                ]
            )
        return lines

    first_context_line = max(1, code_range.start.line - 1)
    last_context_line = min(len(source_lines), code_range.end.line + 1)
    gutter_width = len(str(last_context_line))
    blank_gutter = _line_no_style(f"{' ' * gutter_width} |", color=color)

    lines.append(blank_gutter)
    for line_number in range(first_context_line, last_context_line + 1):
        source_line = source_lines[line_number - 1]
        rendered_source = source_line.expandtabs()
        lines.append(
            f"{_line_no_style(f'{line_number:>{gutter_width}} |', color=color)} {rendered_source}"
        )
        if code_range.start.line <= line_number <= code_range.end.line:
            underline = _render_underline(
                source_line=source_line,
                line_number=line_number,
                code_range=code_range,
                color=color,
            )
            lines.append(f"{blank_gutter} {underline}")

    lines.append(blank_gutter)
    if help_message:
        lines.append(f"{_help_style('help', color=color)}: {help_message}")

    return lines


def _render_underline(
    *, source_line: str, line_number: int, code_range: CodeRange, color: bool
) -> str:
    displayed_line = source_line.expandtabs()

    if code_range.start.line == code_range.end.line:
        start_display = _to_display_column(source_line, code_range.start.column)
        end_display = _to_display_column(source_line, code_range.end.column)
        width = max(1, end_display - start_display)
        return f"{' ' * start_display}{_secondary_code_style('^' * width, color=color)}"

    if line_number == code_range.start.line:
        start_display = _to_display_column(source_line, code_range.start.column)
        width = max(1, len(displayed_line) - start_display)
        return f"{' ' * start_display}{_secondary_code_style('^' * width, color=color)}"

    if line_number == code_range.end.line:
        end_display = _to_display_column(source_line, code_range.end.column)
        width = max(1, end_display)
        return _secondary_code_style("^" * width, color=color)

    width = max(1, len(displayed_line))
    return _secondary_code_style("^" * width, color=color)


def _decode_source_lines(source: FileContent | None) -> list[str]:
    if source is None:
        return []

    return source.decode("utf-8", errors="replace").splitlines()


def _normalize_parser_message(message: str) -> str:
    normalized = _PARSER_ERROR_PREFIX.sub("", message.strip())
    return _PARSER_LOCATION_PREFIX.sub("", normalized)


def _find_syntax_error_end_column(line_text: str, start_column: int) -> int:
    if start_column >= len(line_text):
        return start_column + 1

    if match := re.match(r"[A-Za-z0-9_]+", line_text[start_column:]):
        return start_column + len(match.group(0))

    return start_column + 1


def _to_display_column(source_line: str, column: int) -> int:
    return len(source_line[:column].expandtabs())


def _style(text: str, *, color: bool, **styles: str | bool) -> str:
    if not color:
        return text

    return click.style(text, **styles)


def _error_style(text: str, *, color: bool) -> str:
    return _style(text, color=color, fg="bright_red", bold=True)


def _help_style(text: str, *, color: bool) -> str:
    return _style(text, color=color, fg="bright_cyan", bold=True)


def _line_no_style(text: str, *, color: bool) -> str:
    fg = "bright_cyan" if sys.platform == "win32" else "bright_blue"
    return _style(text, color=color, fg=fg, bold=True)


def _secondary_code_style(text: str, *, color: bool) -> str:
    return _style(text, color=color, fg="red", bold=True)


def _fix_marker(*, color: bool) -> str:
    return f"[{_help_style('*', color=color)}]"
