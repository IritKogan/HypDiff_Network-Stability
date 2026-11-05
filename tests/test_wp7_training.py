import torch
import torch.nn.functional as F

from models.base_models import LPModel
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import generate_synthetic_graph
from stability.wp7_training import _edge_tensor, _loss, model_args


def _model():
    graph = generate_synthetic_graph(71001, "calibration")
    return LPModel(model_args(graph, PilotConfig()))


def test_stable_link_loss_matches_fermi_dirac_binary_cross_entropy():
    model = _model()
    embeddings = torch.tensor(
        [[0.1, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]
    )
    positives = _edge_tensor(((0, 1),), torch.device("cpu"))
    negatives = _edge_tensor(((0, 2),), torch.device("cpu"))
    expected = F.binary_cross_entropy(
        model.decode(embeddings, positives), torch.ones(1)
    ) + F.binary_cross_entropy(model.decode(embeddings, negatives), torch.zeros(1))
    assert torch.allclose(_loss(model, embeddings, positives, negatives), expected)


def test_stable_link_loss_avoids_exponential_overflow():
    model = _model()
    embeddings = torch.tensor(
        [[-0.95, 0.0, 0.0, 0.0], [0.95, 0.0, 0.0, 0.0], [0.0, 0.1, 0.0, 0.0]],
        requires_grad=True,
    )
    positives = _edge_tensor(((0, 1),), torch.device("cpu"))
    negatives = _edge_tensor(((0, 2),), torch.device("cpu"))
    loss = _loss(model, embeddings, positives, negatives)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()
