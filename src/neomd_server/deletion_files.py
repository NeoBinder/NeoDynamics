"""Blocking no-follow inventory and exact owned unlink operations; no database access."""

import hashlib
import json
import os
import stat
import time
from uuid import UUID


class InventoryError(ValueError):
    pass


FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _uuid(value):
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _parent(root, relative):
    parts = relative.split("/")
    if not parts or any(p in {"", ".", ".."} for p in parts) or relative.startswith("/"):
        raise InventoryError("source_changed")
    current = os.dup(root)
    try:
        for name in parts[:-1]:
            child = os.open(name, FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _entry(path, info, session, kind):
    return {
        "session_id": session,
        "resource_kind": "session",
        "resource_id": session,
        "relative_path": path,
        "kind": kind,
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "link_count": info.st_nlink,
    }


def _file_hash(directory, name, info, deadline=None):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
    try:
        digest = hashlib.sha256()
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise InventoryError("inventory_unverified")
            chunk = os.read(fd, 1048576)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ):
            raise InventoryError("source_changed")
        return digest.hexdigest()
    finally:
        os.close(fd)


def inventory_hash(entries):
    # This service-private evidence has exact uint64/ns fields; it is not JCS.
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def scan(storage, sessions, *, timeout, known_aliases=None, maximum_entries=262144):
    deadline = time.monotonic() + timeout
    root = os.open(storage, FLAGS)
    entries, inodes = [], {}
    known_aliases = known_aliases or {}

    def add(entry):
        if len(entries) >= maximum_entries or time.monotonic() >= deadline:
            raise InventoryError("inventory_unverified")
        entries.append(entry)
        if entry["kind"] != "directory":
            inodes.setdefault((entry["device"], entry["inode"]), []).append(entry)

    def walk(directory, path, session, level):
        if level > 64:
            raise InventoryError("inventory_unverified")
        names = []
        with os.scandir(directory) as children:
            for child in children:
                if len(names) + len(entries) >= maximum_entries or time.monotonic() >= deadline:
                    raise InventoryError("inventory_unverified")
                names.append(child.name)
        for name in sorted(names):
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            child_path = path + "/" + name
            if level == 0 and (not _uuid(name) or not stat.S_ISDIR(info.st_mode)):
                raise InventoryError("inventory_unverified")
            if level == 1 and (
                name not in {"output", "checkpoints", "references", "seals", "finalized"}
                or not stat.S_ISDIR(info.st_mode)
            ):
                raise InventoryError("inventory_unverified")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, FLAGS, dir_fd=directory)
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                        raise InventoryError("source_changed")
                    walk(child, child_path, session, level + 1)
                finally:
                    os.close(child)
                add(_entry(child_path, info, session, "directory"))
            elif stat.S_ISREG(info.st_mode):
                add(
                    dict(
                        _entry(child_path, info, session, "file"),
                        content_sha256=_file_hash(directory, name, info, deadline),
                    )
                )
            else:
                raise InventoryError("source_changed")

    try:
        for session in sessions:
            if not _uuid(session):
                raise InventoryError("inventory_unverified")
            try:
                parent, name = _parent(root, "sessions/" + session)
            except FileNotFoundError:
                continue
            try:
                try:
                    directory = os.open(name, FLAGS, dir_fd=parent)
                except FileNotFoundError:
                    continue
                try:
                    info = os.fstat(directory)
                    walk(directory, "sessions/" + session, session, 0)
                    add(_entry("sessions/" + session, info, session, "directory"))
                finally:
                    os.close(directory)
            finally:
                os.close(parent)
        try:
            objects = os.open("objects", FLAGS, dir_fd=root)
        except FileNotFoundError:
            objects = None
        if objects is not None:
            try:
                with os.scandir(objects) as aliases:
                    names = (alias.name for alias in aliases)
                    for name in names:
                        if time.monotonic() >= deadline:
                            raise InventoryError("inventory_unverified")
                        info = os.stat(name, dir_fd=objects, follow_symlinks=False)
                        owned = inodes.get((info.st_dev, info.st_ino))
                        evidence = known_aliases.get(name)
                        if owned is None and evidence is None:
                            continue
                        if not _uuid(name) or not stat.S_ISREG(info.st_mode):
                            raise InventoryError("source_changed")
                        digest = _file_hash(objects, name, info, deadline)
                        if evidence is not None and (
                            info.st_size != evidence["size_bytes"] or digest != evidence["sha256"]
                        ):
                            raise InventoryError("source_changed")
                        session = (
                            evidence["session_id"]
                            if evidence is not None
                            else owned[0]["session_id"]
                        )
                        add(
                            dict(
                                _entry("objects/" + name, info, session, "object_alias"),
                                content_sha256=digest,
                            )
                        )
            finally:
                os.close(objects)
        for paths in inodes.values():
            if paths[0]["link_count"] != len(paths) or any(
                item["session_id"] != paths[0]["session_id"] for item in paths
            ):
                raise InventoryError("inventory_unverified")
        files = [e for e in entries if e["kind"] != "directory"]
        totals = {
            "file_count": len(files),
            "inode_count": len(inodes),
            "exclusive_bytes": sum(paths[0]["size_bytes"] for paths in inodes.values()),
        }
        return entries, totals
    finally:
        os.close(root)


def unlink_entry(storage, entry, inode_paths):
    root = os.open(storage, FLAGS)
    parent = None
    try:
        try:
            parent, name = _parent(root, entry["relative_path"])
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            if not entry["unlink_started"]:
                raise InventoryError("source_changed") from None
            return
        if (info.st_dev, info.st_ino) != (entry["device"], entry["inode"]):
            raise InventoryError("source_changed")
        if entry["kind"] == "directory":
            if not stat.S_ISDIR(info.st_mode):
                raise InventoryError("source_changed")
            os.rmdir(name, dir_fd=parent)  # unexpected new contents refuse removal
        else:
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_size != entry["size_bytes"]
                or info.st_mtime_ns != entry["mtime_ns"]
            ):
                raise InventoryError("source_changed")
            existing = 0
            for other in inode_paths:
                alias_parent = None
                try:
                    alias_parent, alias_name = _parent(root, other["relative_path"])
                    alias = os.stat(alias_name, dir_fd=alias_parent, follow_symlinks=False)
                except FileNotFoundError:
                    if not other["unlink_started"]:
                        raise InventoryError("source_changed") from None
                    continue
                finally:
                    if alias_parent is not None:
                        os.close(alias_parent)
                if (alias.st_dev, alias.st_ino) != (entry["device"], entry["inode"]):
                    raise InventoryError("source_changed")
                existing += 1
            if info.st_nlink != existing:
                raise InventoryError("inventory_unverified")
            if (
                not any(
                    p["unlink_started"] or p["removed"]
                    for p in inode_paths
                    if p["relative_path"] != entry["relative_path"]
                )
                and info.st_ctime_ns != entry["ctime_ns"]
            ):
                raise InventoryError("source_changed")
            if _file_hash(parent, name, info) != entry["content_sha256"]:
                raise InventoryError("source_changed")
            os.unlink(name, dir_fd=parent)
        os.fsync(parent)
    finally:
        if parent is not None:
            os.close(parent)
        os.close(root)
