# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from collections.abc import Collection, Generator, Mapping, Sequence
from dataclasses import replace
from typing import (
    ClassVar,
)

from libcst import (
    BatchableCSTVisitor,
    Comma,
    CSTNode,
    Decorator,
    EmptyLine,
    IndentedBlock,
    Module,
    SimpleStatementSuite,
    TrailingWhitespace,
)
from libcst.metadata import (
    CodePosition,
    CodeRange,
    ParentNodeProvider,
    PositionProvider,
    ProviderT,
)

from .ftypes import (
    Invalid,
    LintIgnoreRegex,
    LintViolation,
    NodeReplacement,
    Valid,
    VisitHook,
    VisitorMethod,
)


class LintRule(BatchableCSTVisitor):
    """
    Lint rule implemented using LibCST.

    To build a new lint rule, subclass this and `Implement a CST visitor
    <https://libcst.readthedocs.io/en/latest/tutorial.html#Build-Visitor-or-Transformer>`_.
    When a lint rule violation should be reported, use the :meth:`report` method.
    """

    METADATA_DEPENDENCIES: ClassVar[Collection[ProviderT]] = (PositionProvider,)
    """
    Required LibCST metadata providers
    """

    TAGS: set[str] = set()
    "Arbitrary classification tags for use in configuration/selection"

    PYTHON_VERSION: str = ""
    """
    Compatible target Python versions, in `PEP 440 version specifier`__ format.

    __ https://peps.python.org/pep-0440/#version-specifiers
    """

    VALID: ClassVar[Sequence[str | Valid]]
    "Test cases that should produce no errors/reports"

    INVALID: ClassVar[Sequence[str | Invalid]]
    "Test cases that are expected to produce errors, with optional replacements"

    AUTOFIX = False  # set by __subclass_init__
    """
    Whether the lint rule contains an autofix.

    Set to ``True`` automatically when :attr:`INVALID` contains at least one
    test case that provides an expected replacment.
    """

    name: str
    """
    Friendly name of this lint rule class, without any "Rule" suffix.
    """

    def __init__(self) -> None:
        self._violations: list[LintViolation] = []
        self.name = self.__class__.__name__
        self.name = self.name.removesuffix("Rule")

    def __init_subclass__(cls) -> None:
        if ParentNodeProvider not in cls.METADATA_DEPENDENCIES:
            cls.METADATA_DEPENDENCIES = (*cls.METADATA_DEPENDENCIES, ParentNodeProvider)

        invalid: list[str | Invalid] = getattr(cls, "INVALID", [])
        for case in invalid:
            if isinstance(case, Invalid) and case.expected_replacement:
                cls.AUTOFIX = True
                return

    def __str__(self) -> str:
        return f"{self.__class__.__module__}:{self.__class__.__name__}"

    _visit_hook: VisitHook | None = None

    def _node_trailing_whitespace(self, node: CSTNode) -> TrailingWhitespace | None:
        trailing_whitespace = getattr(node, "trailing_whitespace", None)
        if trailing_whitespace is not None:
            return trailing_whitespace

        body = getattr(node, "body", None)
        if isinstance(body, SimpleStatementSuite):
            return body.trailing_whitespace
        if isinstance(body, IndentedBlock):
            return body.header
        return None

    def _yield_comment_value(
        self, trailing_whitespace: TrailingWhitespace | None
    ) -> Generator[str, None, None]:
        if trailing_whitespace and trailing_whitespace.comment:
            yield trailing_whitespace.comment.value

    def _yield_empty_line_comments(
        self, empty_lines: Sequence[EmptyLine] | None
    ) -> Generator[str, None, None]:
        if empty_lines is None:
            return

        for line in empty_lines:
            if line.comment:
                yield line.comment.value

    def _yield_direct_node_comments(self, node: CSTNode) -> Generator[str, None, None]:
        yield from self._yield_comment_value(self._node_trailing_whitespace(node))

        comma = getattr(node, "comma", None)
        if isinstance(comma, Comma):
            first_line = getattr(comma.whitespace_after, "first_line", None)
            yield from self._yield_comment_value(first_line)

        right_bracket = getattr(node, "rbracket", None)
        if right_bracket is not None:
            first_line = getattr(right_bracket.whitespace_before, "first_line", None)
            yield from self._yield_comment_value(first_line)

        left_bracket = getattr(node, "lbracket", None)
        if left_bracket is not None:
            yield from self._yield_empty_line_comments(
                getattr(left_bracket.whitespace_after, "empty_lines", None)
            )

        yield from self._yield_empty_line_comments(getattr(node, "lines_after_decorators", None))
        yield from self._yield_empty_line_comments(getattr(node, "leading_lines", None))

    def _should_stop_comment_search(self, node: CSTNode) -> bool:
        return getattr(node, "leading_lines", None) is not None and not isinstance(node, Decorator)

    def node_comments(self, node: CSTNode) -> Generator[str, None, None]:
        """
        Yield all comments associated with the given node.

        Includes comments from both leading comments and trailing inline comments.
        """
        while not isinstance(node, Module):
            yield from self._yield_direct_node_comments(node)
            if self._should_stop_comment_search(node):
                break

            parent = self.get_metadata(ParentNodeProvider, node, None)
            if parent is None:
                break
            node = parent

        # comments at the start of the file are part of the module header rather than
        # part of the first statement's leading_lines, so we need to look there in case
        # the reported node is part of the first statement.
        if isinstance(node, Module):
            for line in node.header:
                if line.comment:
                    yield line.comment.value
        else:
            parent = self.get_metadata(ParentNodeProvider, node, None)
            if isinstance(parent, Module) and parent.body and parent.body[0] == node:
                for line in parent.header:
                    if line.comment:
                        yield line.comment.value

    def ignore_lint(self, node: CSTNode) -> bool:
        """
        Whether to ignore a violation for a given node.

        Returns true if any ``# lint-ignore`` or ``# lint-fixme`` directives match the
        current rule by name, or if the directives have no rule names listed.
        """
        rule_names = (self.name, self.name.lower())
        for comment in self.node_comments(node):
            if match := LintIgnoreRegex.search(comment):
                _style, names = match.groups()

                # directive
                if names is None:
                    return True

                # directive: RuleName
                for name in (n.strip() for n in names.split(",")):
                    name = name.removesuffix("Rule")
                    if name in rule_names:
                        return True

        return False

    def report(
        self,
        node: CSTNode,
        message: str | None = None,
        *,
        position: CodePosition | CodeRange | None = None,
        replacement: NodeReplacement[CSTNode] | None = None,
    ) -> None:
        """
        Report a lint rule violation.

        If `message` is not provided, ``self.MESSAGE`` will be used as a violation
        message. If neither of them are available, this method raises `ValueError`.

        The optional `position` parameter can override the location where the
        violation is reported. By default, the entire span of `node` is used. If
        `position` is a `CodePosition`, only a single character is marked.

        The optional `replacement` parameter can be used to provide an auto-fix for this
        lint violation. Replacing `node` with `replacement` should make the lint
        violation go away.
        """
        if self.ignore_lint(node):
            # TODO: consider logging/reporting this somewhere?
            return

        if not message:
            # backwards compat with Fixit 1.0 api
            message = getattr(self, "MESSAGE", None)
            if not message:
                raise ValueError(f"No message provided in {self.name}")

        if position is None:
            position = self.get_metadata(PositionProvider, node, None)
            if position is None:
                raise ValueError(f"Unable to determine violation position for {self.name}")
        elif isinstance(position, CodePosition):
            end = replace(position, line=position.line + 1, column=0)
            position = CodeRange(start=position, end=end)

        self._violations.append(
            LintViolation(
                self.name,
                range=position,
                message=message,
                node=node,
                replacement=replacement,
            )
        )

    def get_visitors(self) -> Mapping[str, VisitorMethod]:
        def _wrap(name: str, func: VisitorMethod) -> VisitorMethod:
            @functools.wraps(func)
            def wrapper(node: CSTNode) -> None:
                if self._visit_hook:
                    with self._visit_hook(name):
                        return func(node)
                return func(node)

            return wrapper

        return {
            name: _wrap(f"{type(self).__name__}.{name}", visitor)
            for (name, visitor) in super().get_visitors().items()
        }
