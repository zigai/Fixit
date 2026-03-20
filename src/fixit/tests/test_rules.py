# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fixit.config import collect_rules
from fixit.ftypes import Config, QualifiedRule
from fixit.testing import generate_lint_rule_test_cases

for generated_case in generate_lint_rule_test_cases(
    collect_rules(
        Config(
            enable=[
                QualifiedRule("fixit.rules"),
            ],
            python_version=None,
        )
    )
):
    globals()[f"Test{generated_case.__name__}"] = generated_case

del generated_case
