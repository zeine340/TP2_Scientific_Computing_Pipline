from __future__ import annotations

import torch
import torch.nn as nn


# ==========================================================
# Device selection
# ==========================================================

def get_device() -> torch.device:
    """
    Automatically select the best available device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = get_device()


# ==========================================================
# Physics-Informed Neural Network
# ==========================================================

class PINN(nn.Module):
    """
    Simple multilayer perceptron for PINNs.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_hidden_layers: int = 4,
    ):

        super().__init__()

        layers = []

        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x, t):

        inputs = torch.cat([x, t], dim=1)

        return self.network(inputs)


# ==========================================================
# Residual function from Module 3
# ==========================================================

def source_term(
    x,
    t,
    c,
    nu,
):
    """
    Exact source term.
    """

    z = x - c * t

    return 2.0 * nu * torch.tanh(z) * (1.0 - torch.tanh(z) ** 2)


# ==========================================================
# Physics loss
# ==========================================================

def physics_loss(
    model,
    x,
    t,
    c,
    nu,
):

    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    du_dt = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    du_dx = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True,
    )[0]

    residual = (
        du_dt
        + c * du_dx
        - nu * d2u_dx2
        - source_term(x, t, c, nu)
    )

    return torch.mean(residual ** 2)


# ==========================================================
# Boundary condition loss
# ==========================================================

def boundary_loss(
    model,
    xb,
    tb,
    target,
):

    prediction = model(xb, tb)

    return torch.mean((prediction - target) ** 2)


# ==========================================================
# Total loss
# ==========================================================

def total_loss(
    model,
    x,
    t,
    xb,
    tb,
    ub,
    c,
    nu,
):

    l_phys = physics_loss(model, x, t, c, nu)

    l_bc = boundary_loss(model, xb, tb, ub)

    return l_phys + l_bc


# ==========================================================
# Training
# ==========================================================

def train(
    model,
    optimizer,
    epochs,
    x,
    t,
    xb,
    tb,
    ub,
    c,
    nu,
):

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        loss = total_loss(
            model,
            x,
            t,
            xb,
            tb,
            ub,
            c,
            nu,
        )

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch:5d} "
                f"Loss = {loss.item():.6e}"
            )


if __name__ == "__main__":

    model = PINN().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    N = 1000

    x = torch.rand(N, 1, device=DEVICE)
    t = torch.rand(N, 1, device=DEVICE)

    xb = torch.rand(200, 1, device=DEVICE)
    tb = torch.zeros_like(xb)

    ub = torch.tanh(xb)

    train(
        model,
        optimizer,
        epochs=1000,
        x=x,
        t=t,
        xb=xb,
        tb=tb,
        ub=ub,
        c=1.0,
        nu=0.1,
    )