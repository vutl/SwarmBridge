"""Allow-list unpickler for the UAV3D infos files (they reference only numpy + _codecs.encode)."""
import pickle, importlib

ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
           ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
           ("numpy._core.multiarray", "scalar"), ("_codecs", "encode")}


class _Safe(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        return getattr(importlib.import_module(mod), name)


def safe_load(path):
    with open(path, "rb") as f:
        return _Safe(f).load()
