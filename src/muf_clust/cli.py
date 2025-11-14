"""Deprecated CLI shim

Moved to `muf_clust.api.cli`. This module forwards to the new location to
keep backward compatibility for `python -m muf_clust.cli`.
"""

from .api.cli import main

if __name__ == "__main__":
    main()