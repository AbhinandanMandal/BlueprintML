import pytest
import torch
import torch.nn.functional as F

from moco_reproduction.models.moco import MoCo
from test_momentum_update import Tiny


def make(K=8, C=4):
    return MoCo(Tiny(6, C), C, K, 0.99, 0.07)


def keys(n, start, C=4):
    # row i is distinguishable: first coordinate encodes its id
    k = torch.zeros(n, C)
    k[:, 0] = torch.arange(start, start + n, dtype=torch.float)
    return k


def test_initial_queue_is_normalised_and_pointer_zero():
    q = make()
    assert q.queue.shape == (8, 4)
    assert torch.allclose(q.queue.norm(dim=1), torch.ones(8), atol=1e-5)
    assert int(q.queue_ptr) == 0
    assert not q.queue.requires_grad


def test_insertion_order_and_pointer_advance():
    q = make()
    q.enqueue(keys(4, 0))
    assert int(q.queue_ptr) == 4
    assert torch.equal(q.queue[:4, 0], torch.arange(4.0))
    q.enqueue(keys(4, 4))
    assert int(q.queue_ptr) == 0                       # modulo K
    assert torch.equal(q.queue[:, 0], torch.arange(8.0))


def test_fifo_overwrites_oldest():
    q = make()
    q.enqueue(keys(4, 0)); q.enqueue(keys(4, 4)); q.enqueue(keys(4, 8))
    assert int(q.queue_ptr) == 4
    assert torch.equal(q.queue[:4, 0], torch.arange(8.0, 12.0))   # oldest slot replaced
    assert torch.equal(q.queue[4:, 0], torch.arange(4.0, 8.0))


def test_wraparound_when_batch_does_not_divide_queue():
    q = make(K=8)
    q.enqueue(keys(6, 0))
    q.enqueue(keys(6, 6))          # ptr 6 -> fills slots 6,7 then 0..3
    assert int(q.queue_ptr) == 4
    assert torch.equal(q.queue[6:, 0], torch.tensor([6.0, 7.0]))
    assert torch.equal(q.queue[:4, 0], torch.arange(8.0, 12.0))
    assert torch.equal(q.queue[4:6, 0], torch.tensor([4.0, 5.0]))


def test_enqueue_detaches_graph_and_rejects_bad_input():
    q = make()
    k = (torch.randn(4, 4, requires_grad=True) * 2)
    q.enqueue(k)
    assert not q.queue.requires_grad and q.queue.grad_fn is None
    with pytest.raises(ValueError):
        q.enqueue(torch.randn(4, 3))        # wrong dim
    with pytest.raises(ValueError):
        q.enqueue(torch.randn(9, 4))        # larger than queue


def test_queue_and_pointer_are_in_state_dict_and_restore():
    a, b = make(), make()
    a.enqueue(F.normalize(torch.randn(4, 4), dim=1))
    sd = a.state_dict()
    assert "queue" in sd and "queue_ptr" in sd
    b.load_state_dict(sd)
    assert torch.equal(a.queue, b.queue) and int(b.queue_ptr) == 4
