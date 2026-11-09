import torch

from gs_pi.pbr_normalization import PBRNormalFeatureNormalization


def test_mixed_normalization_preserves_legacy_pbr_statistics_and_roundtrip():
    pbr = torch.tensor(
        [
            [0.0, 1.0, 0.25],
            [0.2, 0.8, 0.4],
            [0.1, 0.9, 0.5],
            [0.3, 0.7, 0.6],
            [0.4, 0.6, 0.5],
        ],
        dtype=torch.float32,
    )
    normal = torch.tensor(
        [[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=torch.float32,
    )
    data = torch.cat([pbr, normal], dim=0)
    layer = PBRNormalFeatureNormalization()
    layer.set_from_all_data(data)

    expected_shift = (pbr.min(dim=1).values + pbr.max(dim=1).values) / 2.0
    expected_scale = ((pbr.max(dim=1).values - pbr.min(dim=1).values) / 2.0).max()
    torch.testing.assert_close(layer.shift[:5], expected_shift)
    torch.testing.assert_close(layer.scale[:5], expected_scale.expand(5))
    torch.testing.assert_close(layer.shift[5:], torch.zeros(3))
    torch.testing.assert_close(layer.scale[5:], torch.ones(3))

    batch = data[None, :, :]
    torch.testing.assert_close(layer.denormalize(layer(batch)), batch)
    torch.testing.assert_close(layer(batch)[:, 5:, :], batch[:, 5:, :])


def test_normal_values_do_not_change_pbr_statistics():
    torch.manual_seed(3)
    pbr = torch.rand(5, 31)
    first = PBRNormalFeatureNormalization()
    second = PBRNormalFeatureNormalization()
    first.set_from_all_data(torch.cat([pbr, torch.randn(3, 31)], dim=0))
    second.set_from_all_data(torch.cat([pbr, torch.randn(3, 31) * 1000.0], dim=0))
    torch.testing.assert_close(first.shift[:5], second.shift[:5])
    torch.testing.assert_close(first.scale[:5], second.scale[:5])
