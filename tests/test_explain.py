"""Explainability helpers: exactness of class activation maps and the embryo mask."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("skimage")


def _disc_image(r=70, cx=112, cy=112):
    from PIL import Image, ImageDraw

    im = Image.new("L", (224, 224), 128)
    ImageDraw.Draw(im).ellipse([cx - r, cy - r, cx + r, cy + r], outline=40, width=4)
    return im.convert("RGB")


def test_cam_averages_exactly_to_linear_readout():
    """Mean over the 7x7 grid of the CAM equals the read-out of the pooled features."""
    from headroom import explain as X
    from headroom import images as G

    net = X.load_resnet(None, pretrained=False)
    imgs = [_disc_image(), _disc_image(50, 100, 120)]
    W = np.random.default_rng(0).normal(size=(2, 2048))
    maps = X.cams(net, imgs, W)
    net.to("cpu")
    with torch.no_grad():
        f = net(torch.stack([G._to_tensor(im) for im in imgs])).numpy()
    np.testing.assert_allclose(maps.mean(axis=(2, 3)), f @ W.T, rtol=1e-3, atol=1e-3)


def test_embryo_mask_finds_the_circle():
    from headroom import explain as X

    mask, found = X.embryo_mask(_disc_image(80, 120, 110))
    assert found
    yy, xx = np.mgrid[:224, :224]
    truth = (xx - 120) ** 2 + (yy - 110) ** 2 <= 80 ** 2
    assert (mask & truth).sum() / (mask | truth).sum() > 0.8


def test_enrichment_is_one_for_uniform_map():
    from headroom import explain as X

    mask = np.zeros((224, 224), bool)
    mask[50:150, 50:150] = True
    assert X.enrichment(np.ones((224, 224)), mask) == pytest.approx(1.0)
