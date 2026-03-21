# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect
import logging
import pkgutil
import platform
import sys
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
)

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from .format import FORMAT_STYLES
from .ftypes import (
    Config,
    Options,
    OutputFormat,
    QualifiedRule,
    QualifiedRuleRegex,
    RawConfig,
    RuleOptionsTable,
    T,
    is_rule_option_value,
    is_sequence,
)
from .rule import LintRule
from .util import append_sys_path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

FIXIT_CONFIG_FILENAMES = ("fixit.toml", ".fixit.toml", "pyproject.toml")
FIXIT_LOCAL_MODULE = "fixit.local"


log = logging.getLogger(__name__)


def _find_rules_or_log(qualified_rule: QualifiedRule) -> set[type[LintRule]]:
    try:
        return set(find_rules(qualified_rule))
    except Exception as error:  # noqa: BLE001 - import boundary
        log.warning(
            "Failed to load rules '%s': %s: %s",
            qualified_rule.module,
            error.__class__.__name__,
            error,
        )
        return set()


def _validate_rule(
    rule: str,
    *,
    root: Path,
    config: RawConfig,
    context: str,
    exceptions: list[str],
) -> None:
    try:
        qualified_rule = parse_rule(rule, root, config)
    except Exception as error:  # noqa: BLE001 - validation boundary
        exceptions.append(
            f"Failed to parse rule `{rule}` for {context}: {error.__class__.__name__}: {error}"
        )
        return

    try:
        for _ in find_rules(qualified_rule):
            pass
    except Exception as error:  # noqa: BLE001 - import boundary
        exceptions.append(
            f"Failed to import rule `{rule}` for {context}: {error.__class__.__name__}: {error}"
        )


def _validate_rule_option(
    *,
    rule_name: str,
    settings: Mapping[str, object],
    root: Path,
    config: RawConfig,
    context: str,
) -> str | None:
    try:
        qualified_rule = parse_rule(rule_name, root, config)
    except Exception as error:  # noqa: BLE001 - validation boundary
        return (
            f"Failed to validate options for `{rule_name}` in {context}: "
            f"{error.__class__.__name__}: {error}"
        )

    if qualified_rule.name is None:
        return (
            f"Failed to validate options for `{rule_name}` in {context}: "
            "ConfigError: rule target must reference one concrete rule (`module:ClassName`)"
        )

    try:
        rule_types = list(find_rules(qualified_rule))
    except Exception as error:  # noqa: BLE001 - validation boundary
        return (
            f"Failed to validate options for `{rule_name}` in {context}: "
            f"{error.__class__.__name__}: {error}"
        )

    if len(rule_types) != 1:
        return (
            f"Failed to validate options for `{rule_name}` in {context}: "
            "ConfigError: rule target must resolve to exactly one rule class"
        )

    try:
        rule = rule_types[0]()
        rule.configure(settings)
    except Exception as error:  # noqa: BLE001 - validation boundary
        return (
            f"Failed to validate options for `{rule_name}` in {context}: "
            f"{error.__class__.__name__}: {error}"
        )

    return None


class ConfigError(ValueError):
    def __init__(self, msg: str, config: RawConfig | None = None) -> None:
        super().__init__(msg)
        self.config = config


class CollectionError(RuntimeError):
    def __init__(self, msg: str, rule: QualifiedRule) -> None:
        super().__init__(msg)
        self.rule = rule

    def __reduce__(self) -> tuple[type[RuntimeError], Any]:
        return type(self), (*self.args, self.rule)


def is_rule(obj: type[T]) -> bool:
    """Returns True if class is a concrete subclass of LintRule."""
    return inspect.isclass(obj) and issubclass(obj, LintRule) and obj is not LintRule


@contextmanager
def local_rule_loader(rule: QualifiedRule) -> Iterator[None]:
    """
    Allows importing local rules from arbitrary paths as submodules of fixit.local.

    Imports ``fixit.local``, a "reserved" package within the fixit namespace, and
    overrides the module's path and import spec to come from the root of the specified
    local rule. Relative imports within the local namespace should work correctly,
    though may cause collisions if parent-relative imports (``..foo``) are used.

    When the context exits, this removes all members of the ``fixit.local`` namespace
    from the global ``sys.modules`` dictionary, allowing subsequent imports of further
    local rules.

    This allows importlib to find local names within the fake ``fixit.local`` namespace,
    even if they come from arbitrary places on disk, or would otherwise have namespace
    conflicts if loaded normally using a munged ``sys.path``.
    """
    try:
        import fixit.local

        assert hasattr(fixit.local, "__path__")
        assert fixit.local.__spec__ is not None
        assert rule.root is not None

        orig_spec = fixit.local.__spec__
        fixit.local.__path__ = [rule.root.as_posix()]
        fixit.local.__spec__ = ModuleSpec(
            name=FIXIT_LOCAL_MODULE,
            loader=orig_spec.loader,
            origin=(rule.root / "__init__.py").as_posix(),
            is_package=True,
        )

        yield

    finally:
        for key in list(sys.modules):
            if key.startswith("fixit.local"):
                sys.modules.pop(key, None)


def find_rules(rule: QualifiedRule) -> Iterable[type[LintRule]]:
    """
    Import the rule's qualified module name and return a list of collected rule classes.

    Imports the module by qualified name (eg ``foo.bar`` or ``.local.rules``), and
    then walks that module to find all lint rules.

    If a specific rule name is given, returns only the lint rule matching that name;
    otherwise returns the entire list of found rules.
    """
    try:
        if rule.local:
            with local_rule_loader(rule):
                module = importlib.import_module(rule.module, "fixit.local")
                module_rules = walk_module(module)
        else:
            module = importlib.import_module(rule.module)
            module_rules = walk_module(module)

        if rule.name:
            if value := module_rules.get(rule.name, None):
                if issubclass(value, LintRule):
                    yield value
                else:
                    log.warning("don't know what to do with {value!r}")
            elif rule.local:
                raise CollectionError(f"could not find rule {rule} in {rule.root}", rule)
            else:
                raise CollectionError(f"could not find rule {rule}", rule)

        else:
            for name in sorted(module_rules.keys()):
                yield module_rules[name]

    except ImportError as e:
        if rule.local:
            raise CollectionError(f"could not import rule(s) {rule} from {rule.root}", rule) from e
        else:
            raise CollectionError(f"could not import rule(s) {rule}", rule) from e


def walk_module(module: ModuleType) -> dict[str, type[LintRule]]:
    """
    Given a module object, return a mapping of all rule names to classes.

    Looks at all objects of the module, and collects lint rules that match the
    :func:`is_rule` predicate.

    If the original module is a package (eg, ``foo.__init__``), also loads all
    modules from that package (ignoring sub-packages), and includes their rules in
    the final results.
    """
    rules: dict[str, type[LintRule]] = {}

    members = inspect.getmembers(module, is_rule)
    rules.update(members)

    if hasattr(module, "__path__"):
        for _, module_name, is_pkg in pkgutil.iter_modules(module.__path__):
            if not is_pkg:  # do not recurse to sub-packages
                mod = importlib.import_module(f".{module_name}", module.__name__)
                rules.update(walk_module(mod))

    return rules


def _rule_key_for_type(rule_type: type[LintRule]) -> str:
    return f"{rule_type.__module__}:{rule_type.__name__}"


def _option_key_aliases_for_rule_type(rule_type: type[LintRule]) -> set[str]:
    module_parts = rule_type.__module__.split(".")
    aliases: set[str] = set()
    for idx in range(len(module_parts), 0, -1):
        module_name = ".".join(module_parts[:idx])
        aliases.add(f"{module_name}:{rule_type.__name__}")

    local_prefix = f"{FIXIT_LOCAL_MODULE}."
    if rule_type.__module__.startswith(local_prefix):
        local_module = rule_type.__module__.removeprefix(local_prefix)
        local_parts = local_module.split(".")
        for idx in range(len(local_parts), 0, -1):
            module_name = ".".join(local_parts[:idx])
            aliases.add(f".{module_name}:{rule_type.__name__}")

    return aliases


def collect_rule_types(
    config: Config,
    *,
    # out-param to capture reasons when disabling rules for debugging
    debug_reasons: dict[type[LintRule], str] | None = None,
) -> Collection[type[LintRule]]:
    """Import and return rule types specified by `enables` and `disables`."""
    all_rules: set[type[LintRule]] = set()
    named_enables: set[type[LintRule]] = set()
    disabled_rules = debug_reasons if debug_reasons is not None else {}

    with ExitStack() as stack:
        if config.enable_root_import:
            path = (
                config.root / config.enable_root_import
                if isinstance(config.enable_root_import, Path)
                else config.root
            )
            stack.enter_context(append_sys_path(path))

        for qualified_rule in config.enable:
            rules = _find_rules_or_log(qualified_rule)
            if qualified_rule.name:
                named_enables |= rules
            all_rules |= rules

        for qualified_rule in config.disable:
            disabled_rules.update(
                {
                    rule_type: "disabled"
                    for rule_type in _find_rules_or_log(qualified_rule)
                    if rule_type not in named_enables
                }
            )
            all_rules -= set(disabled_rules)

        if config.tags:
            disabled_rules.update({R: "tags" for R in all_rules if R.TAGS not in config.tags})
            all_rules -= set(disabled_rules)

        if config.python_version is not None:
            disabled_rules.update(
                {
                    R: "python-version"
                    for R in all_rules
                    if R.PYTHON_VERSION
                    and config.python_version
                    not in SpecifierSet(R.PYTHON_VERSION, prereleases=True)
                }
            )
            all_rules -= set(disabled_rules)

    return all_rules


def resolve_rule_settings(
    config: Config,
    rule_types: Collection[type[LintRule]],
) -> dict[type[LintRule], dict[str, object]]:
    rules_by_key: dict[str, type[LintRule]] = {}
    for rule_type in rule_types:
        for alias in _option_key_aliases_for_rule_type(rule_type):
            rules_by_key[alias] = rule_type

    resolved_settings: dict[type[LintRule], dict[str, object]] = {}
    for rule_name, settings in config.options.items():
        rule_type = rules_by_key.get(rule_name)
        if rule_type is None:
            continue

        target_settings = resolved_settings.setdefault(rule_type, {})
        target_settings.update(settings)

    return resolved_settings


def materialize_rules(
    rule_types: Collection[type[LintRule]],
    resolved_settings: Mapping[type[LintRule], Mapping[str, object]],
) -> list[LintRule]:
    materialized_rules: list[LintRule] = []
    for rule_type in sorted(rule_types, key=_rule_key_for_type):
        rule = rule_type()
        rule.configure(resolved_settings.get(rule_type, {}))
        materialized_rules.append(rule)

    return materialized_rules


def collect_rules(
    config: Config,
    *,
    # out-param to capture reasons when disabling rules for debugging
    debug_reasons: dict[type[LintRule], str] | None = None,
) -> Collection[LintRule]:
    """Import, configure, and return rules specified by `enables` and `disables`."""
    rule_types = collect_rule_types(config, debug_reasons=debug_reasons)
    resolved_settings = resolve_rule_settings(config, rule_types)
    return materialize_rules(rule_types, resolved_settings)


def locate_configs(path: Path, root: Path | None = None) -> list[Path]:
    """
    Given a file path, locate all relevant config files in priority order.

    Walking upward from target path, creates a list of candidate paths that exist
    on disk, ordered from nearest/highest priority to further/lowest priority.

    If root is given, only return configs between path and root (inclusive), ignoring
    any paths outside of root, even if they would contain relevant configs.
    If given, root must contain path.

    Returns a list of config paths in priority order, from highest priority to lowest.
    """
    results: list[Path] = []

    if not path.is_dir():
        path = path.parent

    root = root.resolve() if root is not None else Path(path.anchor)
    path.relative_to(root)  # enforce path being inside root

    while True:
        candidates = (path / filename for filename in FIXIT_CONFIG_FILENAMES)
        results.extend(candidate for candidate in candidates if candidate.is_file())

        if path in (root, path.parent):
            break

        path = path.parent

    return results


def read_configs(paths: list[Path]) -> list[RawConfig]:
    """
    Read config data for each path given, and return their raw toml config values.

    Skips any path with no — or empty — `tool.fixit` section.
    Stops early at any config with `root = true`.

    Maintains the same order as given in paths, minus any skipped files.
    """
    configs: list[RawConfig] = []

    for path in paths:
        path = path.resolve()
        content = path.read_text()
        data = tomllib.loads(content)
        fixit_data = data.get("tool", {}).get("fixit", {})

        if fixit_data:
            config = RawConfig(path=path, data=fixit_data)
            configs.append(config)

            if config.data.get("root", False):
                break

    return configs


def get_sequence(
    config: RawConfig, key: str, *, data: dict[str, Any] | None = None
) -> Sequence[str]:
    value: Sequence[str]
    value = data.pop(key, ()) if data else config.data.pop(key, ())

    if not is_sequence(value):
        raise ConfigError(f"{key!r} must be array of values, got {type(value)}", config=config)

    return value


def get_options(  # noqa: C901 - option parsing and normalization
    config: RawConfig, key: str, *, data: dict[str, Any] | None = None
) -> RuleOptionsTable:
    mapping = data.pop(key, {}) if data else config.data.pop(key, {})

    if is_sequence(mapping):
        merged_mapping: dict[object, object] = {}
        for item in mapping:
            if not isinstance(item, Mapping):
                raise ConfigError(
                    f"{key!r} sequence values must be mappings, got {type(item)}",
                    config=config,
                )
            merged_mapping.update(item)
        mapping = merged_mapping

    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{key!r} must be mapping of values, got {type(mapping)}", config=config)

    rule_configs: RuleOptionsTable = {}
    for raw_rule_name, rule_config in mapping.items():
        if not isinstance(raw_rule_name, str):
            raise ConfigError(
                f"{key!r} rule name must be a string, got {type(raw_rule_name)}",
                config=config,
            )

        qualified_rule = parse_rule(raw_rule_name, config.path.parent, config)
        if not qualified_rule.name:
            raise ConfigError(
                f"{key!r} rule target {raw_rule_name!r} must reference one concrete rule (`module:ClassName`)",
                config=config,
            )

        if not isinstance(rule_config, Mapping):
            raise ConfigError(
                f"{key!r} rule config for {raw_rule_name!r} must be a mapping, got {type(rule_config)}",
                config=config,
            )

        rule_name = str(qualified_rule)
        rule_configs[rule_name] = {}
        for option_name, value in rule_config.items():
            if not isinstance(option_name, str):
                raise ConfigError(
                    f"{key!r} option names must be strings, got {type(option_name)}",
                    config=config,
                )

            if not is_rule_option_value(value):
                raise ConfigError(
                    f"{option_name!r} must be a TOML scalar or array of scalars, got {type(value)}",
                    config=config,
                )

            rule_configs[rule_name][option_name] = list(value) if is_sequence(value) else value

    return rule_configs


def parse_rule(rule: str, root: Path, config: RawConfig | None = None) -> QualifiedRule:
    """Given a raw rule string, parse and return a QualifiedRule object."""
    if not (match := QualifiedRuleRegex.match(rule)):
        raise ConfigError(f"invalid rule name {rule!r}", config=config)

    group = match.groupdict()
    module = group["module"]
    name = group["name"]
    local = group["local"]

    if local:
        return QualifiedRule(module, name, local, root)
    return QualifiedRule(module, name)


def merge_configs(  # noqa: C901 - config merge orchestration
    path: Path, raw_configs: list[RawConfig], root: Path | None = None
) -> Config:
    """
    Given multiple raw configs, merge them in priority order.

    Assumes raw_configs are given in order from highest to lowest priority.
    """
    config: RawConfig
    enable_root_import: bool | Path = Config.enable_root_import
    enable_rules: set[QualifiedRule] = {QualifiedRule("fixit.rules")}
    disable_rules: set[QualifiedRule] = set()
    rule_options: RuleOptionsTable = {}
    target_python_version: Version | None = Version(platform.python_version())
    target_formatter: str | None = None
    output_format: OutputFormat = OutputFormat.fixit
    output_template: str = ""

    def update_target_python_version(python_version: str | None) -> None:
        nonlocal target_python_version

        if python_version is None:
            return
        if not isinstance(python_version, str):
            raise ConfigError("'python-version' must be a string", config=config)
        if python_version:
            try:
                target_python_version = Version(python_version)
            except InvalidVersion as error:
                raise ConfigError(
                    f"'python-version' {python_version!r} is not valid",
                    config=config,
                ) from error
            return

        # disable versioning, aka python-version = ""
        target_python_version = None

    def process_subpath(
        subpath: Path,
        *,
        enable: Sequence[str] = (),
        disable: Sequence[str] = (),
        options: RuleOptionsTable | None = None,
        python_version: str | None = None,
        formatter: str | None = None,
    ) -> None:
        nonlocal target_python_version
        nonlocal target_formatter

        subpath = subpath.resolve()
        try:
            path.relative_to(subpath)
        except ValueError:  # not relative to subpath
            return

        config_dir = config.path.parent
        for rule in enable:
            qual_rule = parse_rule(rule, config_dir, config)
            enable_rules.add(qual_rule)
            disable_rules.discard(qual_rule)

        for rule in disable:
            qual_rule = parse_rule(rule, config_dir, config)
            enable_rules.discard(qual_rule)
            disable_rules.add(qual_rule)

        if options:
            for rule_name, option_values in options.items():
                existing_options = rule_options.setdefault(rule_name, {})
                existing_options.update(option_values)

        update_target_python_version(python_version)

        if formatter:
            if formatter not in FORMAT_STYLES:
                raise ConfigError(f"'formatter' {formatter!r} not supported", config=config)

            target_formatter = formatter

    for config in reversed(raw_configs):
        if root is None:
            root = config.path.parent

        data = config.data
        if data.pop("root", False):
            root = config.path.parent

        if value := data.pop("enable-root-import", False):
            if root != config.path.parent:
                raise ConfigError(
                    "enable-root-import not allowed in non-root configs", config=config
                )
            if isinstance(value, str):
                value_path = Path(value)
                if value_path.is_absolute():
                    raise ConfigError(
                        "enable-root-import: absolute paths not allowed", config=config
                    )
                if ".." in value_path.parts:
                    raise ConfigError(
                        "enable-root-import: '..' components not allowed", config=config
                    )
                enable_root_import = value_path
            else:
                enable_root_import = True

        if value := data.pop("output-format", ""):
            try:
                output_format = OutputFormat(value)
            except ValueError as e:
                raise ConfigError("output-format: unknown value {value!r}", config=config) from e

        if value := data.pop("output-template", ""):
            output_template = value

        process_subpath(
            config.path.parent,
            enable=get_sequence(config, "enable"),
            disable=get_sequence(config, "disable"),
            options=get_options(config, "options"),
            python_version=config.data.pop("python-version", None),
            formatter=config.data.pop("formatter", None),
        )

        for override in get_sequence(config, "overrides"):
            if not isinstance(override, dict):
                raise ConfigError("'overrides' requires array of tables", config=config)

            subpath = override.get("path", None)
            if not subpath:
                raise ConfigError("'overrides' table requires 'path' value", config=config)

            subpath = config.path.parent / subpath
            process_subpath(
                subpath,
                enable=get_sequence(config, "enable", data=override),
                disable=get_sequence(config, "disable", data=override),
                options=get_options(config, "options", data=override),
                python_version=override.pop("python-version", None),
                formatter=override.pop("formatter", None),
            )

        for key in data:
            log.warning("unknown configuration option %r", key)

    return Config(
        path=path,
        root=root or Path(path.anchor),
        enable_root_import=enable_root_import,
        enable=sorted(enable_rules),
        disable=sorted(disable_rules),
        options=rule_options,
        python_version=target_python_version,
        formatter=target_formatter,
        output_format=output_format,
        output_template=output_template,
    )


def generate_config(
    path: Path | None = None,
    root: Path | None = None,
    *,
    options: Options | None = None,
) -> Config:
    """Given a file path, walk upwards looking for and applying cascading configs."""
    path = (path or Path.cwd()).resolve()

    if root is not None:
        root = root.resolve()

    if options and options.config_file:
        config_paths = [options.config_file]
    else:
        config_paths = locate_configs(path, root=root)

    raw_configs = read_configs(config_paths)
    config = merge_configs(path, raw_configs, root=root)

    if options:
        if options.tags:
            config.tags = options.tags

        if options.rules:
            config.enable = list(options.rules)
            config.disable = []

        if options.output_format:
            config.output_format = options.output_format

        if options.output_template:
            config.output_template = options.output_template

    return config


def validate_config(path: Path) -> list[str]:  # noqa: C901 - config validation orchestration
    """
    Validate the config provided. The provided path is expected to be a valid toml
    config file. Any exception found while parsing or importing will be added to a list
    of exceptions that are returned.
    """
    exceptions: list[str] = []
    try:
        root = path.parent
        configs = read_configs([path])[0]

        def validate_rules(rules: Sequence[str], context: str) -> None:
            for rule in rules:
                _validate_rule(
                    rule, root=root, config=configs, context=context, exceptions=exceptions
                )

        def validate_rule_options(rule_options: RuleOptionsTable, context: str) -> None:
            for rule_name, settings in rule_options.items():
                error = _validate_rule_option(
                    rule_name=rule_name,
                    settings=settings,
                    root=root,
                    config=configs,
                    context=context,
                )
                if error:
                    exceptions.append(error)

        data = configs.data
        validate_rules(data.get("enable", []), "global enable")
        validate_rules(data.get("disable", []), "global disable")

        try:
            global_options = get_options(configs, "options")
        except Exception as error:  # noqa: BLE001 - validation boundary
            exceptions.append(
                f"Failed to parse options for global options: {error.__class__.__name__}: {error}"
            )
        else:
            validate_rule_options(global_options, "global options")

        for override in data.get("overrides", []):
            if not isinstance(override, dict):
                exceptions.append(
                    "Failed to parse overrides: ConfigError: 'overrides' requires array of tables"
                )
                continue

            override_path = Path(override.get("path", path))
            validate_rules(
                override.get("enable", []),
                f"override enable: `{override_path}`",
            )
            validate_rules(
                override.get("disable", []),
                f"override disable: `{override_path}`",
            )

            try:
                override_options = get_options(configs, "options", data=override)
            except Exception as error:  # noqa: BLE001 - validation boundary
                exceptions.append(
                    f"Failed to parse options for override options: `{override_path}`: {error.__class__.__name__}: {error}"
                )
            else:
                validate_rule_options(override_options, f"override options: `{override_path}`")

    except Exception as error:  # noqa: BLE001 - validation boundary
        exceptions.append(f"Invalid config: {error.__class__.__name__}: {error}")

    return exceptions
