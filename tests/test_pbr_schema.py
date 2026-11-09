import numpy as np
import pytest

from gs_pi.pbr_schema import (
    decode_normal_rgb_numpy,
    get_feature_schema,
    split_feature_channels,
    validate_feature_tensor,
)


def test_legacy_schema_remains_default() -> None:
    assert get_feature_schema().name == "pbr_v1"
    assert get_feature_schema().feats_dim == 5


def test_normal_schema_layout() -> None:
    schema = get_feature_schema("pbr_normal_world_v1")
    assert schema.feats_dim == 8
    assert schema.group("normal").channel_slice == slice(5, 8)
    assert schema.group("normal").space == "blender_world"


def test_split_and_validation() -> None:
    features = np.zeros((2, 8, 17), dtype=np.float32)
    groups = split_feature_channels(features, "pbr_normal_world_v1")
    assert groups["albedo"].shape == (2, 3, 17)
    assert groups["normal"].shape == (2, 3, 17)
    with pytest.raises(ValueError, match="expects 5 feature channels"):
        validate_feature_tensor(features, "pbr_v1")


def test_decode_normal_keeps_zero_invalid() -> None:
    rgb = np.array([[1.0, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    normal, valid, raw_lengths = decode_normal_rgb_numpy(rgb)
    np.testing.assert_allclose(normal[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(normal[1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_allclose(raw_lengths, [0.5, 0.0])


def test_normalization_supports_channel_first_axis() -> None:
    from gs_pi.pbr_schema import normalize_world_normals_numpy

    normals = np.array([[[3.0, 0.0], [0.0, 0.0], [4.0, 0.0]]], dtype=np.float32)
    normalized, valid = normalize_world_normals_numpy(normals, axis=1)
    np.testing.assert_allclose(normalized[0, :, 0], [0.6, 0.0, 0.8])
    np.testing.assert_allclose(normalized[0, :, 1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(valid, [[True, False]])
