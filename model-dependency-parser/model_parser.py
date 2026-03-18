#!/usr/bin/env python3
"""
Generic script to identify all key classes used to build a model file and their paths.

Given a model file (e.g. vLLM's deepseek_v2.py), parses it to find:
  1. Classes defined in the file (with path = the file itself) and their methods
  2. Classes/functions imported from other modules (with resolved filesystem paths)
     and, when the path is known, methods of those classes

Usage:
  python model_build_classes.py <model_file.py> [--root <package_root>] [--format json|text]

Example:
  python model_build_classes.py upstreaming/vllm/vllm/model_executor/models/deepseek_v2.py --root upstreaming/vllm
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path


def is_class_like_name(name: str) -> bool:
    """Heuristic: CamelCase names are typically classes."""
    if not name or name[0].islower():
        return False
    return any(c.isupper() for c in name) or (len(name) > 1 and name[1:].islower())


def resolve_module_to_path(
    module: str, model_file: Path, package_root: Path | None, top_level: str | None = None
) -> str | None:
    """
    Resolve a Python module name to a filesystem path.
    Returns None for stdlib/third-party (e.g. torch, transformers) or if unresolved.
    - Relative imports (.) are resolved relative to model_file's directory.
    - Absolute imports are only resolved when they start with top_level (e.g. 'vllm')
      and package_root is set; otherwise returns None.
    """
    if not package_root or not package_root.is_dir():
        return None
    # Relative imports: .utils, .interfaces -> same dir as model_file
    # (AST sometimes omits the leading dot, so also try same-dir when module has no dot)
    def resolve_relative(mod: str) -> str | None:
        parts = mod.lstrip(".").split(".")
        if not parts or parts[0] == "":
            return str(model_file.parent / "__init__.py")
        base = model_file.parent
        for p in parts[:-1]:
            base = base / p
        name = parts[-1]
        if (base / f"{name}.py").exists():
            return str(base / f"{name}.py")
        if (base / name / "__init__.py").exists():
            return str(base / name / "__init__.py")
        return str(base / f"{name}.py")

    if module.startswith("."):
        return resolve_relative(module)
    # Single-segment module that exists next to model file (AST drops leading dot)
    if "." not in module and (model_file.parent / f"{module}.py").exists():
        return str(model_file.parent / f"{module}.py")
    # Absolute import: only resolve if it belongs to the package (e.g. vllm.*)
    if top_level:
        if not module.startswith(top_level + ".") and module != top_level:
            return None
    parts = module.split(".")
    if not parts:
        return None
    base = package_root
    for p in parts[:-1]:
        base = base / p
    name = parts[-1]
    if (base / f"{name}.py").exists():
        return str(base / f"{name}.py")
    if (base / name / "__init__.py").exists():
        return str(base / name / "__init__.py")
    return str(base / f"{name}.py")


def collect_imported_names(tree: ast.AST) -> list[tuple[str, str]]:
    """Returns list of (imported_name, module)."""
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                out.append((name, module))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                # import foo.bar -> name is "foo" (top-level), module is "foo"
                out.append((name, name))
    return out


def collect_methods_from_class(node: ast.ClassDef) -> list[tuple[str, int]]:
    """Returns list of (method_name, line_number) for direct method definitions."""
    out: list[tuple[str, int]] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((child.name, child.lineno))
    return out


def collect_local_classes_with_methods(
    tree: ast.AST,
) -> list[tuple[str, int, list[tuple[str, int]]]]:
    """Returns list of (class_name, line_number, [(method_name, line), ...])."""
    out: list[tuple[str, int, list[tuple[str, int]]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = collect_methods_from_class(node)
            out.append((node.name, node.lineno, methods))
    return out


def get_class_methods_from_file(
    file_path: Path, class_name: str
) -> list[tuple[str, int]] | None:
    """
    Parse a file and return methods of the first class matching class_name.
    Returns None if file cannot be read/parsed or class not found.
    """
    if not file_path.is_file():
        return None
    try:
        tree = ast.parse(file_path.read_text())
    except (SyntaxError, OSError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return collect_methods_from_class(node)
    return None


def run(
    model_file: Path,
    package_root: Path | None,
    format: str,
    top_level: str | None = None,
) -> None:
    path = model_file.resolve()
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(1)
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        print(f"Error parsing {path}: {e}", file=sys.stderr)
        sys.exit(1)
    root = package_root.resolve() if package_root else None
    # Infer top-level package name from root (e.g. .../vllm -> vllm)
    if root and not top_level:
        top_level = root.name

    # Local classes with methods
    local_classes = collect_local_classes_with_methods(tree)
    # Imported names -> (name, module, resolved_path)
    imports = collect_imported_names(tree)
    key_imports: list[tuple[str, str, str | None]] = []
    for name, module in imports:
        path_str = resolve_module_to_path(module, path, root, top_level)
        # Include class-like names; also include lowercase that are known layer names (e.g. get_rope)
        if is_class_like_name(name) or (module and "vllm" in module and "model_executor" in module):
            key_imports.append((name, module, path_str))

    # Dedupe by name (keep first occurrence)
    seen = set()
    key_imports_dedup = []
    for name, mod, p in key_imports:
        if name not in seen:
            seen.add(name)
            key_imports_dedup.append((name, mod, p))

    # Methods for imported classes (only for class-like names with a path)
    imported_with_methods: list[tuple[str, str, str | None, list[tuple[str, int]] | None]] = []
    for name, mod, path_str in key_imports_dedup:
        methods = None
        if path_str and is_class_like_name(name):
            methods = get_class_methods_from_file(Path(path_str), name)
        imported_with_methods.append((name, mod, path_str, methods))

    if format == "json":
        import json
        result = {
            "model_file": str(path),
            "package_root": str(root) if root else None,
            "local_classes": [
                {
                    "name": n,
                    "line": line,
                    "path": str(path),
                    "methods": [{"name": m, "line": ln} for m, ln in methods_list],
                }
                for n, line, methods_list in local_classes
            ],
            "imported_key_classes": [
                {
                    "name": n,
                    "module": m,
                    "path": p,
                    "methods": [{"name": mn, "line": ln} for mn, ln in (meth or [])],
                }
                for n, m, p, meth in imported_with_methods
            ],
        }
        print(json.dumps(result, indent=2))
    else:
        print("Model file:", path)
        if root:
            print("Package root:", root)
        print()
        print("=== Classes defined in this file ===")
        for name, line, methods_list in local_classes:
            print(f"  {name}  (line {line})  ->  {path}")
            for m, ln in methods_list:
                print(f"      .{m}()  (line {ln})")
        print()
        print("=== Key classes/functions imported (used to build the model) ===")
        for name, module, path_str, methods in imported_with_methods:
            path_display = path_str if path_str else f"[external: {module}]"
            print(f"  {name}  (from {module})  ->  {path_display}")
            if methods is not None:
                for m, ln in methods:
                    print(f"      .{m}()  (line {ln})")


def main() -> None:
    ap = argparse.ArgumentParser(description="List key classes and paths for a model file")
    ap.add_argument("model_file", type=Path, help="Path to the model Python file")
    ap.add_argument("--root", "-r", type=Path, default=None,
                    help="Package root to resolve imports (e.g. upstreaming/vllm)")
    ap.add_argument("--top-level", "-t", type=str, default=None,
                    help="Top-level package name for absolute imports (default: last segment of --root)")
    ap.add_argument("--format", "-f", choices=("text", "json"), default="text",
                    help="Output format")
    args = ap.parse_args()
    run(args.model_file, args.root, args.format, args.top_level)


if __name__ == "__main__":
    main()
