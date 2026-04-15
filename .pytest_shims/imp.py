"""Minimal imp compatibility shim for Python 3.12+ tests."""
import importlib.util, importlib.machinery, sys, types

def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module

def load_source(name, pathname, file=None):
    return load_module(name, pathname)

def new_module(name):
    return types.ModuleType(name)

# Provide legacy names
