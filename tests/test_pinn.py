import torch

from src.deep_pinn import (
    PINN,
    physics_loss,
    get_device,
)


def test_device():

    device = get_device()

    assert device is not None


def test_forward():

    model = PINN()

    x = torch.rand(10, 1)
    t = torch.rand(10, 1)

    y = model(x, t)

    assert y.shape == (10, 1)


def test_physics_loss():

    model = PINN()

    x = torch.rand(20, 1)

    t = torch.rand(20, 1)

    loss = physics_loss(
        model,
        x,
        t,
        c=1.0,
        nu=0.1,
    )

    assert loss.item() >= 0