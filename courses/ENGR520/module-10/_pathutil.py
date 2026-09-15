"""module-10 -- _pathutil.py

Both module-3 and module-9 have their own `frames.py` (different modules,
same filename -- "Body Frame vs Inertial Frame" in Module 3, the full
C->E->L->B pipeline backbone in Module 9). Adding both directories to
sys.path and doing `from frames import ...` is ambiguous: Python's module
cache is keyed by name, so whichever one resolves first "wins" for the rest
of the process, silently, depending on import order.

load_module() sidesteps that by loading a specific file under a private
alias, bypassing sys.path/sys.modules name resolution entirely for that one
file. Used only for the file whose name collides (module-9's frames.py);
everything else (e.g. module-3's quaternions.py) has no name clash and is
imported the normal way.
"""

import importlib.util


def load_module(alias, file_path):
    spec = importlib.util.spec_from_file_location(alias, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
