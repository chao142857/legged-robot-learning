"""Process-local observer bootstrap; no simulator/package files are modified."""
import os
import sys

if os.environ.get("NATIVE_WBC_DIAGNOSTICS"):
    import functools
    import importlib.abc
    import importlib.machinery

    class _ObserverLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            native_learn = module.OnPolicyRunner.learn

            @functools.wraps(native_learn)
            def observed_learn(runner, *args, **kwargs):
                from native_observer import install
                install(runner)
                return native_learn(runner, *args, **kwargs)

            module.OnPolicyRunner.learn = observed_learn

    class _ObserverFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "rsl_rl.runners.on_policy_runner":
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                spec.loader = _ObserverLoader(spec.loader)
                return spec
            return None

    sys.meta_path.insert(0, _ObserverFinder())
