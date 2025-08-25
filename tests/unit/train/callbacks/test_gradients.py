import torch
import types

from rslearn.train.callbacks.gradients import MiniPCGrad


def test_minipcgrad():
    """
    Test that the MiniPCGrad callback projects conflicting gradients to be orthogonal.
    """

    # Dummy module with two parameters; only one matches the selector and is affected.
    class DummyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # param to be processed by callback (selector 'target')
            self.target_weight = torch.nn.Parameter(torch.zeros(2))
            # param that should be ignored
            self.other_weight = torch.nn.Parameter(torch.zeros(2))

        def forward(self):  # not used
            pass

    pl_module = DummyModule()

    # Named parameters must include the selector for the target param
    def named_params():
        return [("target.weight", pl_module.target_weight), ("other.weight", pl_module.other_weight)]
    pl_module.named_parameters = types.MethodType(lambda self: named_params(), pl_module)

    cb = MiniPCGrad(selector="target", only_monitor=False)

    # Monkeypatch logging on the callback (Callback doesn't own log_dict by default)
    logs = []
    cb.log_dict = lambda d, **_: logs.append(d)

    # Simulate lightning's batch start to set dataset_source/batch_size
    batch = ([{"dataset_source": "train"}],)  # matches access: batch[0][0]["dataset_source"]
    cb.on_train_batch_start(trainer=None, pl_module=pl_module, batch=batch, batch_idx=0)

    # Reset prev grads at optimizer step start
    cb.on_before_optimizer_step(trainer=None, pl_module=pl_module, optimizer=None)

    # First micro-batch gradient (prev = g1)
    g1 = torch.tensor([1.0, 0.0])
    pl_module.target_weight.grad = g1.clone()
    pl_module.other_weight.grad = torch.tensor([0.5, 0.5])  # should be ignored
    cb.on_after_backward(trainer=None, pl_module=pl_module)

    # Second micro-batch arrives; gradients are accumulated -> grad = g1 + g2
    g2 = torch.tensor([-1.0, 1.0])  # conflicts with g1 (dot = -1)
    accumulated = g1 + g2
    pl_module.target_weight.grad = accumulated.clone()
    cb.on_after_backward(trainer=None, pl_module=pl_module)

    # After projection, the micro part (new_grad - prev_grad) should be orthogonal to prev.
    # Retrieve prev from cb.prev_grads (it stores the latest param.grad)
    final_grad = cb.prev_grads["target.weight"][0]
    prev_grad = g1  # from first micro-batch (before projection step)
    new_micro = final_grad - prev_grad

    # Check orthogonality: dot(prev, new_micro) = 0 (within tolerance)
    dot_val = torch.dot(prev_grad.flatten(), new_micro.flatten()).item()
    assert abs(dot_val) < 1e-6, f"Projected micro-grad not orthogonal: dot={dot_val}, prev={prev_grad}, micro'={new_micro}"
    assert torch.allclose(pl_module.other_weight.grad, torch.tensor([0.5, 0.5])), \
        "Non-selected parameter's gradient should not be modified"
