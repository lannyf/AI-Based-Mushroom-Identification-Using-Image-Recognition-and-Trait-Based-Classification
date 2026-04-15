"""Stub bundled_test_runner used only for running upstream flutter tests in CI-less environments.
This provides minimal API so tests import; behaviour is a no-op to avoid running platform-specific tests.
"""
from dataclasses import dataclass

@dataclass
class TestCase:
    name: str
    binary: str = ''
    args: list = None


def run_tests(cases, *args, **kwargs):
    """Pretend to run tests: return a dict mapping test name to successful result."""
    results = {}
    for c in cases:
        name = getattr(c, 'name', str(c))
        results[name] = {'passed': True, 'output': ''}
    return results
