# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Triggers built-in lint rules
class Foo:
    def bar(self, value: str) -> str:
        return f"value is {value}"


# lint-fixme: SomethingUnrelated
class Bar:
    pass


# lint-ignore
class Phi:
    pass


# lint-fixme: NoInheritFromObject
class Rho:
    pass


class Zeta:  # lint-ignore NoInheritFromObject
    pass
