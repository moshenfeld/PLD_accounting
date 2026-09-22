"""Verify documented package-root exports and overview-named helpers exist."""

from __future__ import annotations

import ast
import dataclasses
import importlib
from pathlib import Path

import PLD_accounting


# Public symbols listed in IMPLEMENTATION_OVERVIEW / public README that an
# installed user imports from the package root or the documented module path.
_PACKAGE_ROOT_NAMES = (
    "compose_full_pld",
    "discrete_distribution",
    "dp_accounting_pmf_to_pld_realization",
    "gaussian_allocation_delta_configurable",
    "gaussian_allocation_directional_pld",
    "gaussian_allocation_epsilon_configurable",
    "gaussian_allocation_epsilon_range",
    "gaussian_allocation_pld",
    "gaussian_distribution",
    "general_allocation_delta",
    "general_allocation_epsilon",
    "general_allocation_pld",
    "GridSpec",
    "laplace_distribution",
    "rediscretize_dist_by_bound",
    "subsample_pld",
    "subsample_pld_realization",
)

_MODULE_PATH_HELPERS = (
    (
        "PLD_accounting.random_allocation_gaussian",
        "_embed_positive_boundary_on_nonpositive_real_cell",
    ),
    (
        "PLD_accounting.random_allocation_gaussian",
        "_fold_nonpositive_real_mass_to_positive_boundary",
    ),
    ("PLD_accounting.geometric_convolution", "_add_single_zero_atom_cross_term"),
)

_JIT_DECORATOR_NAMES = frozenset({"njit", "jit", "vectorize", "optional_njit"})
# NumPy fallback twin of a positional Numba kernel; both keep one call shape.
_POSITIONAL_KERNEL_TWINS = frozenset(
    {
        ("distribution_discretization.py", "_numpy_rediscretize_prob"),
    }
)


def _decorator_attr_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        current = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(current, ast.Name):
            names.add(current.id)
        elif isinstance(current, ast.Attribute):
            names.add(current.attr)
    return names


def _has_positional_multi_input_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Return whether a multi-input signature leaves any input positional."""
    positional = [
        arg.arg
        for arg in (*node.args.posonlyargs, *node.args.args)
        if arg.arg not in {"self", "cls"}
    ]
    keyword_only = [arg.arg for arg in node.args.kwonlyargs]
    return bool(positional) and len(positional) + len(keyword_only) > 1


def test_documented_root_exports_are_in_all() -> None:
    """Every documented root export is in ``__all__``, and every ``__all__`` name resolves.

    A subset check in both directions, not an equality: ``__all__`` also carries types
    and constants that the documented-callable list deliberately omits.
    """
    for name in PLD_accounting.__all__:
        assert hasattr(PLD_accounting, name), name
    for name in _PACKAGE_ROOT_NAMES:
        assert name in PLD_accounting.__all__, name
        assert callable(getattr(PLD_accounting, name)) or name == "GridSpec"


def test_overview_named_helpers_exist_at_documented_paths() -> None:
    """Overview-named helpers remain importable at their documented module paths."""
    assert hasattr(PLD_accounting.GridSpec, "geometric")
    for module_name, attr in _MODULE_PATH_HELPERS:
        module = importlib.import_module(module_name)
        assert callable(getattr(module, attr)), f"{module_name}.{attr}"


def test_public_dataclasses_with_several_fields_are_keyword_only() -> None:
    """A public dataclass carrying more than one field is constructed by keyword."""
    package_root = Path(PLD_accounting.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        stem = path.relative_to(package_root).with_suffix("").as_posix().replace("/", ".")
        module_name = "PLD_accounting" if stem == "__init__" else f"PLD_accounting.{stem}"
        module = importlib.import_module(module_name)
        for name, obj in vars(module).items():
            if name.startswith("_") or not isinstance(obj, type):
                continue
            # Re-exports belong to the module that defines them.
            if not dataclasses.is_dataclass(obj) or obj.__module__ != module_name:
                continue
            fields = dataclasses.fields(obj)
            if len(fields) > 1 and not all(field.kw_only for field in fields):
                offenders.append(f"{module_name}.{name}")
    assert not offenders, "multi-field public dataclasses must be kw_only: " + ", ".join(
        sorted(set(offenders))
    )


def test_custom_constructors_with_several_inputs_are_keyword_only() -> None:
    """Custom constructors with several inputs put every input after ``*``."""
    package_root = Path(PLD_accounting.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name == "__init__" and _has_positional_multi_input_signature(node):
                offenders.append(f"{path.relative_to(package_root)}:{node.lineno}")
    assert not offenders, "multi-input custom constructors must be keyword-only: " + ", ".join(
        offenders
    )


def test_public_multi_input_functions_are_keyword_only() -> None:
    """Public functions with several inputs use keywords."""
    package_root = Path(PLD_accounting.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        relative_path = path.relative_to(package_root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            if _has_positional_multi_input_signature(node):
                offenders.append(f"{relative_path}:{node.lineno}:{node.name}")
    assert not offenders, "multi-input public functions must be keyword-only: " + ", ".join(
        offenders
    )


def test_private_multi_input_helpers_are_keyword_only() -> None:
    """Private helpers with several inputs use keywords except JIT kernels and twins."""
    package_root = Path(PLD_accounting.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        relative_path = path.relative_to(package_root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("_") or node.name.startswith("__"):
                continue
            if _decorator_attr_names(node) & _JIT_DECORATOR_NAMES:
                continue
            exception_key = (relative_path.as_posix(), node.name)
            if exception_key in _POSITIONAL_KERNEL_TWINS:
                continue
            if _has_positional_multi_input_signature(node):
                offenders.append(f"{relative_path}:{node.lineno}:{node.name}")
    assert not offenders, "multi-input private helpers must be keyword-only: " + ", ".join(
        offenders
    )


def test_keyword_only_check_rejects_a_mixed_signature() -> None:
    """One positional plus one keyword-only input still violates the convention."""
    node = ast.parse("def example(first, *, second):\n    pass\n").body[0]
    assert isinstance(node, ast.FunctionDef)
    assert _has_positional_multi_input_signature(node)


def test_private_helpers_declare_no_default_arguments() -> None:
    """A default belongs on a public entry point or a constant, not on a ``_name`` helper."""
    package_root = Path(PLD_accounting.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Dunder and protocol methods keep the signature the protocol requires.
            if not node.name.startswith("_") or node.name.startswith("__"):
                continue
            args = node.args
            if args.defaults or any(default is not None for default in args.kw_defaults):
                offenders.append(f"{path.relative_to(package_root)}:{node.lineno}:{node.name}")
    assert not offenders, "private helpers must not declare defaults; found: " + ", ".join(
        offenders
    )


def test_single_input_functions_are_positional() -> None:
    """A leading ``*`` is only for functions with more than one input argument."""
    package_root = Path(PLD_accounting.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            positional = [
                arg.arg for arg in (*args.posonlyargs, *args.args) if arg.arg not in {"self", "cls"}
            ]
            keyword_only = [arg.arg for arg in args.kwonlyargs]
            if args.vararg is not None:
                continue
            if not positional and len(keyword_only) == 1:
                rel = path.relative_to(package_root)
                offenders.append(f"{rel}:{node.lineno}:{node.name}")
    assert (
        not offenders
    ), "single-input functions must stay positional; found keyword-only *: " + ", ".join(offenders)
