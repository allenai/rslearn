"""Test gradient surgery."""

import types

import torch

from rslearn.train.callbacks.gradients import MiniPCGrad


def test_minipcgrad() -> None:
    """
    Test that MiniPCGrad projects conflicting gradients to be orthogonal.
    """

    class Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proc_weight = torch.nn.Parameter(torch.zeros(2))
            self.other_weight = torch.nn.Parameter(torch.zeros(2))
            self.proc_blocked_weight = torch.nn.Parameter(torch.zeros(2))

        def forward(self) -> None:
            pass

    m = Dummy()

    # Stable names for selection logic
    def named_params() -> list[tuple[str, torch.nn.Parameter]]:
        return [
            ("proc.weight", m.proc_weight),
            ("other.weight", m.other_weight),
            ("proc_blocked.weight", m.proc_blocked_weight),
        ]

    m.named_parameters = types.MethodType(lambda self: named_params(), m)

    cb = MiniPCGrad(selectors=["proc"], deselectors=["blocked"], only_monitor=False)
    cb.log_dict = lambda d, **kw: None  # monkeypatch logging

    # Simulate Lightning calls
    batch = ([{"dataset_source": "train"}],)
    cb.on_train_batch_start(None, m, batch, 0)
    cb.on_before_optimizer_step(None, m, None)

    # First micro-batch
    g1_proc = torch.tensor([1.0, 0.0])
    g1_other = torch.tensor([1.0, 0.0])
    g1_blocked = torch.tensor([1.0, 0.0])
    m.proc_weight.grad = g1_proc.clone()
    m.other_weight.grad = g1_other.clone()
    m.proc_blocked_weight.grad = g1_blocked.clone()
    cb.on_after_backward(None, m)

    # Second micro-batch: conflict for proc param
    g2_proc = torch.tensor([-1.0, 1.0])  # conflicts with g1_proc
    g2_other = torch.tensor([-1.0, 1.0])  # ignored
    g2_blocked = torch.tensor([-1.0, 1.0])  # ignored
    m.proc_weight.grad = (g1_proc + g2_proc).clone()
    m.other_weight.grad = (g1_other + g2_other).clone()
    m.proc_blocked_weight.grad = (g1_blocked + g2_blocked).clone()
    cb.on_after_backward(None, m)

    # PROCESSED param must have orthogonal micro-grad
    final_grad = cb.prev_grads["proc.weight"][0]
    prev = g1_proc
    new_micro = final_grad - prev
    dot_val = torch.dot(prev.flatten(), new_micro.flatten()).item()
    assert abs(dot_val) < 1e-6, f"Proc param not projected correctly: dot={dot_val}"

    # SKIPPED params must remain as naive sums
    assert torch.allclose(m.other_weight.grad, g1_other + g2_other), (
        "Non-selector param should be skipped"
    )
    assert torch.allclose(m.proc_blocked_weight.grad, g1_blocked + g2_blocked), (
        "Deselected param should be skipped"
    )
