# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    get_args,
    get_origin,
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


class RuleConfigurationError(ValueError):
    pass


_RULE_SETTING_MISSING = object()
_SCALAR_SETTING_TYPES = (str, int, float, bool)


def _is_scalar_setting_type(value: object) -> bool:
    return isinstance(value, type) and value in _SCALAR_SETTING_TYPES


def _is_instance_for_type(value: object, expected: type[object]) -> bool:
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return type(value) is int
    if expected is float:
        return type(value) is float
    if expected is str:
        return isinstance(value, str)
    return isinstance(value, expected)


@dataclass(frozen=True)
class RuleSetting:
    value_type: object
    default: object = _RULE_SETTING_MISSING
    validator: Callable[[object], object] | None = None

    def _validate_type(
        self,
        *,
        value: object,
        setting_name: str,
        rule_name: str,
    ) -> None:
        expected_type = self.value_type
        origin = get_origin(expected_type)
        if origin is list:
            args = get_args(expected_type)
            if len(args) != 1 or not _is_scalar_setting_type(args[0]):
                raise RuleConfigurationError(
                    f"{rule_name}: unsupported list type for setting {setting_name!r}: {expected_type!r}"
                )

            item_type = args[0]
            if not isinstance(value, list):
                raise RuleConfigurationError(
                    f"{rule_name}: setting {setting_name!r} expected {expected_type!r}, got {type(value)!r}"
                )

            if not all(_is_instance_for_type(item, item_type) for item in value):
                raise RuleConfigurationError(
                    f"{rule_name}: setting {setting_name!r} expected items of type {item_type!r}, got {value!r}"
                )
            return

        if not _is_scalar_setting_type(expected_type):
            raise RuleConfigurationError(
                f"{rule_name}: unsupported type for setting {setting_name!r}: {expected_type!r}"
            )

        if not _is_instance_for_type(value, expected_type):
            raise RuleConfigurationError(
                f"{rule_name}: setting {setting_name!r} expected {expected_type!r}, got {type(value)!r}"
            )

    def validate(
        self,
        value: object,
        *,
        setting_name: str,
        rule_name: str,
    ) -> object:
        self._validate_type(value=value, setting_name=setting_name, rule_name=rule_name)

        if self.validator:
            try:
                validator_result = self.validator(value)
            except Exception as error:
                raise RuleConfigurationError(
                    f"{rule_name}: setting {setting_name!r} failed validation: {error}"
                ) from error

            if validator_result is False:
                raise RuleConfigurationError(
                    f"{rule_name}: setting {setting_name!r} failed validation"
                )

        return value


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

    SETTINGS: ClassVar[dict[str, RuleSetting]] = {}
    "Optional typed configuration settings for this lint rule."

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
        self.settings: Mapping[str, Any] = MappingProxyType({})
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
        return self.qualified_name()

    @classmethod
    def qualified_name(cls) -> str:
        return f"{cls.__module__}:{cls.__name__}"

    def configure(self, raw_settings: Mapping[str, object]) -> None:
        unknown_settings = sorted(set(raw_settings) - set(self.SETTINGS))
        if unknown_settings:
            available = sorted(self.SETTINGS)
            raise RuleConfigurationError(
                f"{self.qualified_name()}: unknown setting(s) {unknown_settings!r}; available settings: {available!r}"
            )

        resolved_settings: dict[str, object] = {}
        for setting_name, setting in self.SETTINGS.items():
            if setting_name in raw_settings:
                value = raw_settings[setting_name]
            elif setting.default is _RULE_SETTING_MISSING:
                raise RuleConfigurationError(
                    f"{self.qualified_name()}: missing required setting {setting_name!r}"
                )
            else:
                value = deepcopy(setting.default)

            resolved_settings[setting_name] = setting.validate(
                value,
                setting_name=setting_name,
                rule_name=self.qualified_name(),
            )

        self.settings = MappingProxyType(resolved_settings)

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
        message: str,
        *,
        position: CodePosition | CodeRange | None = None,
        replacement: NodeReplacement[CSTNode] | None = None,
    ) -> None:
        """
        Report a lint rule violation.

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
