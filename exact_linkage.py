import os
import pathlib

def _install_pyximport():
    # Make sure build artifacts go somewhere writable/persistent-ish.
    build_dir = pathlib.Path(
        os.environ.get("PYXIMPORT_BUILD_DIR", pathlib.Path.cwd() / ".pyximport")
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np  
    import pyximport

    os.environ.setdefault("CC", "cc")
    os.environ.setdefault("CXX", "c++")
    os.environ["CFLAGS"] = (os.environ.get("CFLAGS", "") + " -pthread").strip()
    os.environ["LDFLAGS"] = (os.environ.get("LDFLAGS", "") + " -pthread").strip()
    # Sometimes distutils uses LDSHARED explicitly
    os.environ.setdefault("LDSHARED", "cc -shared")

    # IMPORTANT: include_dirs must include NumPy headers.
    pyximport.install(
        build_dir=str(build_dir),
        inplace=False,
        language_level=3,
        setup_args={
            "include_dirs": [np.get_include()],
            # If you need extra compile/link flags, add them here, e.g.:
            # "extra_compile_args": ["-O3"],
        },
    )

def _import_exact_linkage_cython():
    """
    Import the compiled Cython module, building it on first import.
    """
    try:
        from .exact_linkage_cython import nn_chain
        return nn_chain
    except ImportError:
        _install_pyximport()
        from .exact_linkage_cython import nn_chain
        return nn_chain

def exact_linkage(dists, n, corr):
    nn_chain = _import_exact_linkage_cython()
    return nn_chain(dists, n, corr)
