"""Directory-entry durability for immutable files in an owned Linux PVC root."""

import os
from pathlib import Path


def sync_directories(directory, *, root):
    sync_directory_trees([directory], root=root)


def sync_directory_trees(directories, *, root):
    """Sync changed trees and common ancestors once, after all file mutations."""
    root, paths = Path(root).resolve(), set()
    for value in directories:
        directory = Path(value).resolve()
        if not directory.is_relative_to(root):
            raise ValueError("directory is outside its owned storage root")
        paths.update({directory, *(path for path in directory.rglob("*") if path.is_dir())})
        parent = directory
        while parent != root:
            parent = parent.parent
            paths.add(parent)
    for path in sorted(paths, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
