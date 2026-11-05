import torch

from models.base_models import LPModel
from stability.pilot_config import PilotConfig
from stability.synthetic_instances import generate_synthetic_graph
from stability.wp7_training import model_args


def _model(final_activation="relu"):
    graph = generate_synthetic_graph(71001, "calibration")
    args = model_args(
        graph, PilotConfig(), final_activation=final_activation
    )
    return LPModel(args)


def test_final_activation_defaults_to_relu_and_leaky_relu_is_opt_in():
    relu_model = _model()
    leaky_model = _model("leaky_relu")
    negative_input = torch.tensor([-1.0, 1.0])

    assert torch.equal(
        relu_model.encoder.layers[0].hyp_act.act(negative_input),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(
        relu_model.encoder.layers[1].hyp_act.act(negative_input),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(
        leaky_model.encoder.layers[0].hyp_act.act(negative_input),
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(
        leaky_model.encoder.layers[1].hyp_act.act(negative_input),
        torch.tensor([-0.01, 1.0]),
    )
