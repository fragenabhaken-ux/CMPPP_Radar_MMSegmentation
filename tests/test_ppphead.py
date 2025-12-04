import torch
import pytest
import numpy as np

from mmseg.models.decode_heads import PPPHead


@pytest.fixture
def head():
    """Create a PPPHead for testing."""
    return PPPHead(
        in_channels=512,
        channels=128,
        num_convs=2,
        num_classes=1,
        loss_ppp=dict(type='PPPLoss')
    )


# ------------------------------------------------------------------------------
# TEST 1 — extract_centers_from_label (1 instance)
# ------------------------------------------------------------------------------
def test_extract_centers_from_label_single(head):
    gt = torch.zeros(1,1,64,128, dtype=torch.int64)
    gt[:,:,20:30,40:50] = 26000

    centers = head.extract_centers_from_label(gt)
    assert len(centers) == 1
    assert centers[0].shape == (1,2)
    assert torch.allclose(centers[0][0], torch.tensor([44.5, 24.5], device=gt.device))


# ------------------------------------------------------------------------------
# TEST 2 — extract_centers_from_label (multiple instances)
# ------------------------------------------------------------------------------
def test_extract_centers_from_label_multiple(head):
    gt = torch.zeros(1,1,64,128, dtype=torch.int64)

    # car instance 0
    gt[:,:,10:20,20:30] = 26000
    # car instance 1
    gt[:,:,30:40,70:90] = 26001
    # person instance 0
    gt[:,:,45:50,10:15] = 24000

    centers = head.extract_centers_from_label(gt)[0]

    assert centers.shape[0] == 3


# ------------------------------------------------------------------------------
# TEST 3 — centers_to_mask
# ------------------------------------------------------------------------------
def test_centers_to_mask(head):
    centers = [torch.tensor([[44.5, 24.5]])]
    logits = torch.zeros(1,64,128)

    mask = head.centers_to_mask(centers, logits)

    assert mask.sum() == 1
    assert mask[0,24,44] == 1


# ------------------------------------------------------------------------------
# TEST 4 — forward pass
# ------------------------------------------------------------------------------
def test_forward_pass(head):
    feat = torch.randn(1,512,64,128)
    out = head([feat])

    assert out.shape == (1,1,64,128)


# ------------------------------------------------------------------------------
# TEST 5 — PPP loss end-to-end
# ------------------------------------------------------------------------------
def test_ppp_loss(head):
    feat = torch.randn(1,512,64,128)
    logits = head([feat])

    gt = torch.zeros(1,1,64,128, dtype=torch.int64)
    gt[:,:,20:30,40:50] = 26000

    losses = head.loss(logits, gt)

    assert "loss_ppp" in losses
    assert torch.is_tensor(losses["loss_ppp"])


# ------------------------------------------------------------------------------
# TEST 6 — no instances (empty image)
# ------------------------------------------------------------------------------
def test_no_instances(head):
    feat = torch.randn(1,512,64,128)
    logits = head([feat])

    gt = torch.zeros(1,1,64,128, dtype=torch.int64)  # no instances

    losses = head.loss(logits, gt)

    assert torch.is_tensor(losses["loss_ppp"])


# ------------------------------------------------------------------------------
# TEST 7 — batch testing (different number of instances)
# ------------------------------------------------------------------------------
def test_batch_processing(head):
    gt = torch.zeros(2,1,64,128, dtype=torch.int64)

    # image 1 → 1 instance
    gt[0,:,20:30,40:50] = 26000
    # image 2 → 3 instances
    gt[1,:, 5:15,  5:15] = 26000
    gt[1,:,30:40, 70:90] = 26001
    gt[1,:,45:55, 20:30] = 24000

    feat = torch.randn(2,512,64,128)
    logits = head([feat])

    losses = head.loss(logits, gt)
    assert "loss_ppp" in losses


# ------------------------------------------------------------------------------
# TEST 8 — extract centers from prediction (debug)
# ------------------------------------------------------------------------------
def test_extract_centers_from_prediction(head):
    feat = torch.randn(1,512,64,128)
    logits = head([feat]).clone()

    # introduce a strong peak
    logits[:,0,20:30,40:50] += 10

    pred_centers = head.extract_centers_from_prediction(logits)

    assert len(pred_centers[0]) >= 1
