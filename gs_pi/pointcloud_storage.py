"""Legacy view PLYs and compact, lossless per-view access to the final PLY.

Only an explicit storage marker enables virtual views. An incomplete legacy
object must not silently become a compact object. No point is reordered here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData

STORAGE_MARKER = "_POINTCLOUD_STORAGE.json"
STORAGE_FORMAT = "gs_pi_compact_pointcloud_v1"
PHASE1_STORAGE_FORMAT = "gs_pi_phase1_pointcloud_v1"
DUAL_STORAGE_FORMAT = "gs_pi_exact_and_2dgs_views_v1"
LEAN_PRESERVED_PROFILE = "lean_preserved_v1"
LEAN_PREVIEW_PATHS = tuple(
    f"extract_output/previews/three_way_view16/{name}_view16.png"
    for name in ("GT_blender", "GS_fill_view_source", "mesh_exact_view_source")
)
PHASE1_PLY = "merged_model_render_color.ply"
PHASE1_VIEW_FIELDS = ("x", "y", "z", "albedo_red", "albedo_green", "albedo_blue")
FINAL_PLY = "extract_output/merged_model_render_color.ply"
FINAL_FIELDS = {
    "x",
    "y",
    "z",
    "red",
    "green",
    "blue",
    "metallic",
    "roughness",
    "albedo_red",
    "albedo_green",
    "albedo_blue",
    "view_source",
    "nx",
    "ny",
    "nz",
    "normal_valid",
    "normal_gt_source_view",
}


def _object_root(base_dir):
    path = Path(base_dir)
    return path.parent if path.name == "extract_output" else path


def _read_vertices(path):
    from plyfile import PlyParseError

    try:
        return PlyData.read(str(path), mmap="c")["vertex"].data
    except (PlyParseError, KeyError) as exc:
        raise ValueError(f"invalid vertex PLY: {path}: {exc}") from exc


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _view_index(vertices, num_views, required_fields=FINAL_FIELDS):
    if num_views < 1 or len(vertices) == 0:
        raise ValueError("compact point cloud requires positive view and point counts")
    if not required_fields.issubset(vertices.dtype.names or ()):
        raise ValueError(
            "compact point cloud is missing final PBR/normal/view_source fields"
        )
    source = vertices["view_source"]
    if (
        source.dtype.kind not in "iu"
        or np.any(source < 1)
        or np.any(source > num_views)
    ):
        raise ValueError(
            "compact view_source must contain integer IDs in [1, num_views]"
        )
    if np.any(source[1:] < source[:-1]):
        raise ValueError("compact points must retain the generator's per-view order")
    counts = np.bincount(source.astype(np.int64), minlength=num_views + 1)
    offset, views = 0, []
    for view_id in range(1, num_views + 1):
        count = int(counts[view_id])
        views.append({"view_id": view_id, "start_idx": offset, "count": count})
        offset += count
    return views


def validate_compact_object(object_root, *, num_views=None, verify_hash=False):
    root = _object_root(object_root)
    marker = json.loads((root / STORAGE_MARKER).read_text())
    if not isinstance(marker, dict) or marker.get("format") != STORAGE_FORMAT:
        raise ValueError(f"invalid compact storage marker: {root}")
    if marker.get("point_cloud_relpath") != FINAL_PLY:
        raise ValueError(f"compact marker must reference {FINAL_PLY}")
    count = marker.get("num_views")
    if (
        type(count) is not int
        or count < 1
        or (num_views is not None and count != num_views)
    ):
        raise ValueError(f"compact marker view count mismatch: {root}")
    path = root / FINAL_PLY
    vertices = _read_vertices(path)
    views = _view_index(vertices, count)
    if (
        marker.get("point_count") != len(vertices)
        or marker.get("ply_bytes") != path.stat().st_size
        or marker.get("views") != views
    ):
        raise ValueError(f"compact marker does not match final PLY: {root}")
    if verify_hash and marker.get("ply_sha256") != _sha256(path):
        raise ValueError(f"compact final PLY SHA-256 mismatch: {root}")
    return marker, vertices


def validate_phase1_storage(object_root, *, num_views=None, verify_hash=False):
    """Validate explicitly indexed Phase1 data; do not accept incomplete legacy data."""
    root = _object_root(object_root)
    marker = json.loads((root / STORAGE_MARKER).read_text())
    if (
        marker.get("format") != PHASE1_STORAGE_FORMAT
        or marker.get("point_cloud_relpath") != PHASE1_PLY
        or marker.get("view_fields") != list(PHASE1_VIEW_FIELDS)
    ):
        raise ValueError(f"invalid Phase1 storage marker: {root}")
    count = marker.get("num_views")
    if (
        type(count) is not int
        or count < 1
        or (num_views is not None and count != num_views)
    ):
        raise ValueError(f"invalid Phase1 view count: {root}")
    path = root / PHASE1_PLY
    vertices = _read_vertices(path)
    views = _view_index(vertices, count, FINAL_FIELDS - {"normal_gt_source_view"})
    info = json.loads((root / "pointcloud_info.json").read_text())
    if (
        marker.get("views") != views
        or info.get("views") != views
        or marker.get("point_count") != len(vertices)
        or info.get("total_points") != len(vertices)
        or marker.get("ply_bytes") != path.stat().st_size
    ):
        raise ValueError(f"Phase1 view indices differ from merged PLY: {root}")
    if verify_hash and marker.get("ply_sha256") != _sha256(path):
        raise ValueError(f"Phase1 PLY SHA-256 mismatch: {root}")
    return marker, vertices


def finalize_phase1_storage(object_root, *, num_views=32):
    """Publish virtual-view metadata for newly generated --skip-view-ply objects."""
    root = Path(object_root)
    path = root / PHASE1_PLY
    vertices = _read_vertices(path)
    views = _view_index(vertices, num_views, FINAL_FIELDS - {"normal_gt_source_view"})
    info = json.loads((root / "pointcloud_info.json").read_text())
    if info.get("views") != views or info.get("total_points") != len(vertices):
        raise ValueError(f"Phase1 pointcloud_info does not match merged PLY: {root}")
    marker = {
        "format": PHASE1_STORAGE_FORMAT,
        "point_cloud_relpath": PHASE1_PLY,
        "num_views": num_views,
        "point_count": len(vertices),
        "views": views,
        "view_fields": list(PHASE1_VIEW_FIELDS),
        "ply_bytes": path.stat().st_size,
        "ply_sha256": _sha256(path),
    }
    _write_json_atomic(root / STORAGE_MARKER, marker)
    return marker


def _fields_copy(vertices, fields):
    out = np.empty(
        len(vertices), dtype=[(k, vertices.dtype.fields[k][0]) for k in fields]
    )
    for key in fields:
        out[key] = vertices[key]
    return out


def validate_dual_view_storage(
    object_root, *, prefer_extract_output=True, num_views=None, verify_hash=False
):
    """Validate independently indexed exact and reconstructed clouds."""
    root = _object_root(object_root)
    marker = json.loads((root / STORAGE_MARKER).read_text())
    if not isinstance(marker, dict) or marker.get("format") != DUAL_STORAGE_FORMAT:
        raise ValueError(f"invalid dual-view storage marker: {root}")
    count = marker.get("num_views")
    if (
        type(count) is not int
        or count < 1
        or (num_views is not None and count != num_views)
    ):
        raise ValueError(f"dual-view count mismatch: {root}")
    chosen = None
    for key, relpath, fields in (
        ("exact", PHASE1_PLY, list(PHASE1_VIEW_FIELDS)),
        ("reconstructed", FINAL_PLY, None),
    ):
        record = marker.get("clouds", {}).get(key, {})
        if (
            record.get("point_cloud_relpath") != relpath
            or record.get("view_fields") != fields
        ):
            raise ValueError(f"dual-view cloud layout mismatch: {key}")
        path = root / relpath
        vertices = _read_vertices(path)
        required = (
            FINAL_FIELDS - {"normal_gt_source_view"} if key == "exact" else FINAL_FIELDS
        )
        views = _view_index(vertices, count, required)
        if (
            record.get("views") != views
            or record.get("point_count") != len(vertices)
            or record.get("ply_bytes") != path.stat().st_size
        ):
            raise ValueError(f"dual-view index does not match {key} PLY: {root}")
        if verify_hash and record.get("ply_sha256") != _sha256(path):
            raise ValueError(f"dual-view {key} PLY SHA-256 mismatch: {root}")
        if (key == "reconstructed") == bool(prefer_extract_output):
            chosen = (
                {**record, "format": DUAL_STORAGE_FORMAT, "num_views": count},
                vertices,
            )
    return chosen


def finalize_dual_view_storage(object_root, *, num_views=32):
    """Verify both sets of legacy views, publish indices, then remove view PLYs.

    The exact merge and final merge remain separate, byte-for-byte unchanged.
    This is explicitly selected by the preserve-attributes lean output mode.
    """
    root = Path(object_root)
    marker = {"format": DUAL_STORAGE_FORMAT, "num_views": num_views, "clouds": {}}
    redundant = []
    for key, relpath, fields in (
        ("exact", PHASE1_PLY, list(PHASE1_VIEW_FIELDS)),
        ("reconstructed", FINAL_PLY, None),
    ):
        path = root / relpath
        vertices = _read_vertices(path)
        required = (
            FINAL_FIELDS - {"normal_gt_source_view"} if key == "exact" else FINAL_FIELDS
        )
        views = _view_index(vertices, num_views, required)
        for view in views:
            legacy_path = path.parent / f"view_{view['view_id']:02d}.ply"
            if not legacy_path.exists():
                continue
            legacy = _read_vertices(legacy_path)
            start = view["start_idx"]
            expected = vertices[start : start + view["count"]]
            if fields is not None:
                expected = _fields_copy(expected, fields)
            if legacy.dtype != expected.dtype or not np.array_equal(legacy, expected):
                raise ValueError(f"view differs from its {key} merge: {legacy_path}")
            redundant.append(legacy_path)
        marker["clouds"][key] = {
            "point_cloud_relpath": relpath,
            "view_fields": fields,
            "views": views,
            "point_count": len(vertices),
            "ply_bytes": path.stat().st_size,
            "ply_sha256": _sha256(path),
        }
    _write_json_atomic(root / STORAGE_MARKER, marker)
    validate_dual_view_storage(root, num_views=num_views, verify_hash=True)
    for path in redundant:
        path.unlink()
    return marker


def lean_preserved_output_ready(object_root):
    root = _object_root(object_root)
    if any(
        not (root / path).is_file() or (root / path).stat().st_size == 0
        for path in LEAN_PREVIEW_PATHS
    ):
        return False
    if (root / "final_scene_debug.blend").exists():
        return False
    for directory in (root, root / "extract_output"):
        if any(directory.glob("view_*.ply")):
            return False
    if any(
        (root / "extract_output" / name).exists()
        for name in ("2dgs_surface_raw.ply", "2dgs_surface_filtered.ply")
    ):
        return False
    return {
        p.relative_to(root).as_posix()
        for p in (root / "extract_output/previews").rglob("*.png")
    } == set(LEAN_PREVIEW_PATHS)


def prune_lean_preserved_debug_assets(object_root):
    """Remove only the explicitly excluded debug assets after validation."""
    root = _object_root(object_root)
    validate_dual_view_storage(root, verify_hash=True)
    if any(
        not (root / path).is_file() or (root / path).stat().st_size == 0
        for path in LEAN_PREVIEW_PATHS
    ):
        raise ValueError("Generate all three retained previews before pruning")
    for name in (
        "final_scene_debug.blend",
        "extract_output/2dgs_surface_raw.ply",
        "extract_output/2dgs_surface_filtered.ply",
    ):
        (root / name).unlink(missing_ok=True)
    for path in (root / "extract_output/previews").rglob("*.png"):
        if path.relative_to(root).as_posix() not in LEAN_PREVIEW_PATHS:
            path.unlink()
    if not lean_preserved_output_ready(root):
        raise ValueError(f"Incomplete lean preserved output: {root}")


class ViewPointCloudReader:
    """Read legacy files first; reuse one merged PLY mapping for virtual views."""

    def __init__(self, base_dir, prefer_extract_output=None):
        root = _object_root(base_dir)
        if prefer_extract_output is True:
            self.directories = [root / "extract_output"]
        elif prefer_extract_output is False:
            self.directories = [root]
        else:
            self.directories = [root / "extract_output", root]
        self.root = root
        self._compact = None

    def _legacy_path(self, view_id):
        for directory in self.directories:
            path = directory / f"view_{view_id:02d}.ply"
            if path.is_file() and path.stat().st_size > 0:
                return path
        return None

    def read(self, view_id):
        if type(view_id) is not int or view_id < 1:
            raise ValueError("view_id must be a positive integer")
        path = self._legacy_path(view_id)
        if path is not None:
            return _read_vertices(path)
        if (self.root / STORAGE_MARKER).is_file():
            if self._compact is None:
                layout = json.loads((self.root / STORAGE_MARKER).read_text()).get(
                    "format"
                )
                if layout == PHASE1_STORAGE_FORMAT and self.root in self.directories:
                    self._compact = validate_phase1_storage(self.root)
                elif (
                    layout == STORAGE_FORMAT
                    and self.root / "extract_output" in self.directories
                ):
                    self._compact = validate_compact_object(self.root)
                elif layout == DUAL_STORAGE_FORMAT:
                    self._compact = validate_dual_view_storage(
                        self.root,
                        prefer_extract_output=self.directories[0] != self.root,
                    )
                else:
                    raise FileNotFoundError(
                        f"view storage not available for requested layout: {self.root}"
                    )
            marker, vertices = self._compact
            if view_id <= marker["num_views"]:
                view = marker["views"][view_id - 1]
                start = view["start_idx"]
                chunk = vertices[start : start + view["count"]]
                if marker["format"] == PHASE1_STORAGE_FORMAT or marker.get(
                    "view_fields"
                ) == list(PHASE1_VIEW_FIELDS):
                    # Legacy Phase1 view PLYs contain XYZ + float albedo, not
                    # the merged PLY's quantized rendered RGB. Preserve this.
                    return _fields_copy(chunk, PHASE1_VIEW_FIELDS)
                return chunk
        raise FileNotFoundError(f"missing view {view_id}: {self.root}")

    def missing_views(self, num_views):
        missing = []
        for view_id in range(1, num_views + 1):
            if self._legacy_path(view_id) is not None:
                continue
            try:
                self.read(view_id)
            except FileNotFoundError:
                missing.append(view_id)
        return missing


def _write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def finalize_compact_object(object_root, *, num_views=32):
    """Finalize a newly generated/private object, then remove redundant assets.

    Never call this on a published legacy object in place. Production calls it
    only after successful extraction, before publishing the success marker/tar.
    Existing complete datasets are skipped by the generation entry point.
    """
    root = Path(object_root)
    path = root / FINAL_PLY
    vertices = _read_vertices(path)
    views = _view_index(vertices, num_views)
    # Resumed extraction can have legacy outputs. Prove they are redundant
    # before removing them; root Phase1 PLYs are intermediate inputs instead.
    for view in views:
        legacy_path = root / "extract_output" / f"view_{view['view_id']:02d}.ply"
        if legacy_path.exists():
            legacy = _read_vertices(legacy_path)
            start = view["start_idx"]
            if legacy.dtype != vertices.dtype or not np.array_equal(
                legacy, vertices[start : start + view["count"]]
            ):
                raise ValueError(f"view differs from final PLY: {legacy_path}")
    duplicate = root / "2DGS_point.ply"
    if duplicate.exists() and _sha256(duplicate) != _sha256(path):
        raise ValueError(f"2DGS_point.ply differs from final PLY: {root}")
    info_path = root / "extract_output/pointcloud_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    info.update(total_points=len(vertices), views=views)
    _write_json_atomic(info_path, info)
    normal_manifest_path = root / "extract_output/normal_manifest.json"
    if normal_manifest_path.exists():
        normal_manifest = json.loads(normal_manifest_path.read_text())
        normal_manifest.update(
            after_normal_previews=[], previews=[], pointcloud_storage=STORAGE_FORMAT
        )
        _write_json_atomic(normal_manifest_path, normal_manifest)
    marker = {
        "format": STORAGE_FORMAT,
        "point_cloud_relpath": FINAL_PLY,
        "num_views": num_views,
        "point_count": len(vertices),
        "views": views,
        "ply_bytes": path.stat().st_size,
        "ply_sha256": _sha256(path),
    }
    # Publish the layout before pruning, so interruption leaves a readable
    # object. Object-level _SUCCESS is still published by the production gate.
    _write_json_atomic(root / STORAGE_MARKER, marker)
    redundant = [
        root / "2DGS_point.ply",
        root / "merged_model_render_color.ply",
        root / "pointcloud_info.json",
        root / "final_scene_debug.blend",
    ]
    for directory in (root, root / "extract_output"):
        redundant.extend(
            directory / f"view_{i:02d}.ply" for i in range(1, num_views + 1)
        )
    for item in redundant:
        item.unlink(missing_ok=True)
    for name in ("hole_fill_debug", "previews"):
        directory = root / "extract_output" / name
        if directory.is_symlink():
            directory.unlink()
        elif directory.exists():
            shutil.rmtree(directory)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate"])
    parser.add_argument("object_root", type=Path)
    parser.add_argument("--num-views", type=int, default=32)
    args = parser.parse_args()
    marker, _ = validate_compact_object(
        args.object_root, num_views=args.num_views, verify_hash=True
    )
    print(
        json.dumps({"format": marker["format"], "point_count": marker["point_count"]})
    )


if __name__ == "__main__":
    main()
