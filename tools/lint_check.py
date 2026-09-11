#!/usr/bin/env python3
"""Simple AST-based linter for unused imports and undefined names."""

import ast
import sys
from pathlib import Path


class LintChecker(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.imported_names = set()
        self.defined_names = set()
        self.used_names = set()
        self.builtins = {
            'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set',
            'tuple', 'bool', 'type', 'isinstance', 'enumerate', 'zip', 'map',
            'filter', 'sum', 'min', 'max', 'abs', 'sorted', 'reversed', 'all',
            'any', 'open', 'file', 'input', 'iter', 'next', 'object', 'Exception',
            'ValueError', 'TypeError', 'KeyError', 'IndexError', 'AttributeError',
            'RuntimeError', 'NotImplementedError', 'StopIteration', 'BaseException',
        }
        self.issues = []

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.imported_names.add((name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name != '*':
                name = alias.asname if alias.asname else alias.name
                self.imported_names.add((name, node.lineno))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defined_names.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def check(self):
        for name, lineno in self.imported_names:
            if name not in self.used_names and name not in self.defined_names:
                self.issues.append((self.filename, lineno, f'F401: {name} imported but unused'))

        for name in self.used_names:
            if name not in self.imported_names and name not in self.defined_names and name not in self.builtins:
                self.issues.append((self.filename, 0, f'F821: {name} undefined'))


def lint_file(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content, filepath)
        checker = LintChecker(filepath)
        checker.visit(tree)
        checker.check()
        return checker.issues
    except SyntaxError as e:
        return [(filepath, e.lineno, f'SyntaxError: {e.msg}')]
    except Exception as e:
        return [(filepath, 0, f'Error: {e}')]


def main():
    repo_root = Path(__file__).parent.parent
    py_files = list(repo_root.glob('src/**/*.py')) + list(repo_root.glob('tests/**/*.py'))

    all_issues = []
    for filepath in sorted(py_files):
        issues = lint_file(str(filepath))
        all_issues.extend(issues)

    if all_issues:
        for filepath, lineno, message in all_issues:
            if lineno:
                print(f'{filepath}:{lineno}: {message}')
            else:
                print(f'{filepath}: {message}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
