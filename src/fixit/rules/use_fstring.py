# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections.abc import Callable
from typing import cast

import libcst as cst
import libcst.matchers as m

from fixit import Invalid, LintRule, RuleSetting, Valid

USE_FSTRING_SIMPLE_EXPRESSION_MAX_LENGTH = 30


def _match_simple_string(node: cst.BaseExpression) -> bool:
    if isinstance(node, cst.SimpleString) and "b" not in node.prefix.lower():
        # SimpleString can be a bytes and fstring don't support bytes
        # This helper is for autofixer and it only handles %s simple cases for now.
        # for the other types like %f %d, it require more processing on the format.
        # Convert a %d to fstring will change the exception type. We may get help
        # from TypeInferenceProvider to confirm the type before codedmod.
        # We leave it as future work.
        return re.fullmatch("[^%]*(%s[^%]*)+", node.raw_value) is not None
    return False


def _gen_match_simple_expression(
    codegen: Callable[[cst.CSTNode], str],
    simple_expression_max_length: int,
) -> Callable[[cst.BaseExpression], bool]:
    def _match_simple_expression(node: cst.BaseExpression) -> bool:
        # either each element in Tuple is simple expression or the entire expression is simple.
        return bool(
            (
                isinstance(node, cst.Tuple)
                and all(
                    len(codegen(elm.value)) < simple_expression_max_length for elm in node.elements
                )
            )
            or len(codegen(node)) < simple_expression_max_length
        )

    return _match_simple_expression


class EscapeStringQuote(cst.CSTTransformer):
    def __init__(self, quote: str) -> None:
        self.quote = quote
        super().__init__()

    def leave_SimpleString(
        self, original_node: cst.SimpleString, _updated_node: cst.SimpleString
    ) -> cst.SimpleString:
        if self.quote == original_node.quote:
            for quo in ["'", '"', "'''", '"""']:
                if quo != original_node.quote and quo not in original_node.raw_value:
                    escaped_string = cst.SimpleString(
                        original_node.prefix + quo + original_node.raw_value + quo
                    )
                    if escaped_string.evaluated_value != original_node.evaluated_value:
                        raise ValueError(
                            f"Failed to escape string:\n  original:{original_node.value}\n  escaped:{escaped_string.value}"
                        )
                    return escaped_string
            raise ValueError(
                f"Cannot find a good quote for escaping the SimpleString: {original_node.value}"
            )
        return original_node


class UseFstring(LintRule):
    """
    Encourages the use of f-string instead of %-formatting or .format() for high code quality and efficiency.

    Following two cases not covered:

    1. arguments length greater than 30 characters: for better readibility reason
        For example:

        1: this is the answer: %d" % (a_long_function_call() + b_another_long_function_call())
        2: f"this is the answer: {a_long_function_call() + b_another_long_function_call()}"
        3: result = a_long_function_call() + b_another_long_function_call()
        f"this is the answer: {result}"

        Line 1 is more readable than line 2. Ideally, we'd like developers to manually fix this case to line 3

    2. only %s placeholders are linted against for now. We leave it as future work to support other placeholders.
        For example, %d raises TypeError for non-numeric objects, whereas f"{x:d}" raises ValueError.
        This discrepancy in the type of exception raised could potentially break the logic in the code where the exception is handled
    """

    MESSAGE: str = (
        "Do not use printf style formatting or .format(). "
        "Use f-string instead to be more readable and efficient. "
        "See https://www.python.org/dev/peps/pep-0498/"
    )
    SETTINGS = {
        "simple_expression_max_length": RuleSetting(
            int,
            default=USE_FSTRING_SIMPLE_EXPRESSION_MAX_LENGTH,
        ),
    }

    VALID = [
        Valid("somebody='you'; f\"Hey, {somebody}.\""),
        Valid('"hey"'),
        Valid('"hey" + "there"'),
        Valid('b"a type %s" % var'),
    ]

    INVALID = [
        Invalid('"Hey, {somebody}.".format(somebody="you")'),
        Invalid('"%s" % "hi"', expected_replacement='''f"{'hi'}"'''),
        Invalid('"a name: %s" % name', expected_replacement='f"a name: {name}"'),
        Invalid(
            '"an attribute %s ." % obj.attr',
            expected_replacement='f"an attribute {obj.attr} ."',
        ),
        Invalid(
            'r"raw string value=%s" % val',
            expected_replacement='fr"raw string value={val}"',
        ),
        Invalid('"{%s}" % val', expected_replacement='f"{{{val}}}"'),
        Invalid('"{%s" % val', expected_replacement='f"{{{val}"'),
        Invalid(
            '"The type of var: %s" % type(var)',
            expected_replacement='f"The type of var: {type(var)}"',
        ),
        Invalid(
            '"%s" % obj.this_is_a_very_long_expression(parameter)["a_very_long_key"]',
        ),
        Invalid(
            '"%s" % abcdefghijklmnopqrstuvwxyz1234567890',
            expected_replacement='f"{abcdefghijklmnopqrstuvwxyz1234567890}"',
            options={"simple_expression_max_length": 100},
        ),
        Invalid(
            '"type of var: %s, value of var: %s" % (type(var), var)',
            expected_replacement='f"type of var: {type(var)}, value of var: {var}"',
        ),
        Invalid(
            "'%s\" double quote is used' % var",
            expected_replacement="f'{var}\" double quote is used'",
        ),
        Invalid(
            '"var1: %s, var2: %s, var3: %s, var4: %s" % (class_object.attribute, dict_lookup["some_key"], some_module.some_function(), var4)',
            expected_replacement='''f"var1: {class_object.attribute}, var2: {dict_lookup['some_key']}, var3: {some_module.some_function()}, var4: {var4}"''',
        ),
        Invalid(
            '"a list: %s" % " ".join(var)',
            expected_replacement='''f"a list: {' '.join(var)}"''',
        ),
    ]

    _codegen: Callable[[cst.CSTNode], str] | None

    def visit_Module(self, node: cst.Module) -> bool | None:
        self._codegen = node.code_for_node
        return super().visit_Module(node)

    def visit_Call(self, node: cst.Call) -> None:
        if m.matches(
            node,
            m.Call(func=m.Attribute(value=m.SimpleString(), attr=m.Name(value="format"))),
        ):
            self.report(node, self.MESSAGE)

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        codegen = self._codegen
        if not codegen:
            raise ValueError("No codegen found. Have we visited a Module?")
        simple_expression_max_length = self.settings["simple_expression_max_length"]
        assert isinstance(simple_expression_max_length, int)

        expr_key = "expr"
        extracts = m.extract(
            node,
            m.BinaryOperation(
                left=m.MatchIfTrue(_match_simple_string),
                operator=m.Modulo(),
                right=m.SaveMatchedNode(
                    m.MatchIfTrue(
                        _gen_match_simple_expression(codegen, simple_expression_max_length)
                    ),
                    expr_key,
                ),
            ),
        )

        if extracts:
            expr = cast(cst.BaseExpression, extracts[expr_key])
            parts: list[cst.BaseFormattedStringContent] = []
            simple_string = cst.ensure_type(node.left, cst.SimpleString)
            innards = simple_string.raw_value.replace("{", "{{").replace("}", "}}")
            tokens = innards.split("%s")
            token = tokens[0]
            if len(token) > 0:
                parts.append(cst.FormattedStringText(value=token))
            expressions = (
                [elm.value for elm in expr.elements] if isinstance(expr, cst.Tuple) else [expr]
            )
            escape_transformer = EscapeStringQuote(simple_string.quote)
            i = 1
            while i < len(tokens):
                if i - 1 >= len(expressions):
                    # Only generate warning for cases where %-string not comes with same number of elements in tuple
                    self.report(node, self.MESSAGE)
                    return
                try:
                    parts.append(
                        cst.FormattedStringExpression(
                            expression=cast(
                                cst.BaseExpression,
                                # pyre-fixme[16]: `Sequence` has no attribute `visit`.
                                expressions[i - 1].visit(escape_transformer),
                            )
                        )
                    )
                except ValueError:
                    self.report(node, self.MESSAGE)
                    return
                token = tokens[i]
                if len(token) > 0:
                    parts.append(cst.FormattedStringText(value=token))
                i += 1
            start = f"f{simple_string.prefix}{simple_string.quote}"
            replacement = cst.FormattedString(parts=parts, start=start, end=simple_string.quote)
            self.report(node, self.MESSAGE, replacement=replacement)
        elif m.matches(
            node, m.BinaryOperation(left=m.SimpleString(), operator=m.Modulo())
        ) and isinstance(cst.ensure_type(node.left, cst.SimpleString).evaluated_value, str):
            self.report(node, self.MESSAGE)
