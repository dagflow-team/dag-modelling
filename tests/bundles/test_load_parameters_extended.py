from tempfile import NamedTemporaryFile

from numpy import abs, allclose, float64
from pytest import mark
from yaml import safe_dump

from dag_modelling.bundles.load_parameters import load_parameters

# TODO: add test for offset


STATE_FORMAT_to_NS = {
    ("variable", ("value", "sigma_percent")): "constrained",
    ("variable", ("value", "sigma_relative")): "constrained",
    ("variable", ("value", "sigma_absolute")): "constrained",
    ("variable", "value"): "free",
    ("fixed", "value"): "constant",
}

replicate_indices = [(), ("0", "1", "2")]


def _get_parameter_value_label(parameters, labels, key_sequence, replicate_indices):
    parameter = parameters.copy()
    label = labels.copy()
    for key in key_sequence:
        if key in replicate_indices:
            break
        parameter = parameter[key]
        label = label[key]
    return parameter, label


def _load_parameters_from_file(format, state, parameters, labels, replicate_indices):
    with NamedTemporaryFile(suffix=".yaml") as f:
        filename = f.name
        with open(filename, "w") as fin:
            safe_dump(dict(format=format, state=state, parameters=parameters, labels=labels), fin)

        storage = load_parameters(
            path="",
            load=filename,
            replicate=replicate_indices,
        )
    return storage


def _load_parameters_from_dict(format, state, parameters, labels, replicate_indices):
    return load_parameters(
        format=format,
        state=state,
        parameters=parameters,
        labels=labels,
        replicate=replicate_indices,
    )


@mark.parametrize("state", ["variable", "fixed"])
@mark.parametrize("format", ["value"])
@mark.parametrize(
    "parameters,labels",
    [
        ({"x": 1}, {"x": "Label for x"}),
        ({"nested": {"x": -1.5, "y": 1.5}}, {"nested": {"x": "Label for x", "y": "Label for y"}}),
    ],
)
@mark.parametrize("replicate_indices", replicate_indices)
@mark.parametrize("load_from_file", [True, False])
def test_load_parameters(state, format, parameters, labels, replicate_indices, load_from_file):
    if load_from_file:
        storage = _load_parameters_from_file(format, state, parameters, labels, replicate_indices)
    else:
        storage = _load_parameters_from_dict(format, state, parameters, labels, replicate_indices)

    for key, loaded_parameter in storage("parameters.all").walkitems():
        parameter, label = _get_parameter_value_label(parameters, labels, key, replicate_indices)

        loaded_value = loaded_parameter.to_dict()["value"]
        loaded_label = loaded_parameter.to_dict()["label"]

        assert parameter == loaded_value, "Loaded value is not equal to initial"
        assert label == loaded_label, "Loaded label is not equal to initial"
        assert len(storage(f"parameters.{STATE_FORMAT_to_NS[(state, format)]}").keys()) > 0


@mark.parametrize("sigma_format", ["sigma_percent", "sigma_relative", "sigma_absolute"])
@mark.parametrize(
    "parameters,labels",
    [
        ({"x": (1, 1)}, {"x": "Label for x"}),
        (
            {"nested": {"x": (-1.5, 0.1), "y": (1.5, 0.1)}},
            {"nested": {"x": "Label for x", "y": "Label for y"}},
        ),
    ],
)
@mark.parametrize("replicate_indices", replicate_indices)
@mark.parametrize("load_from_file", [True, False])
def test_load_parameters_constrained(
    sigma_format, parameters, labels, replicate_indices, load_from_file
):
    state = "variable"
    fmt = ("value", sigma_format)

    if load_from_file:
        storage = _load_parameters_from_file(fmt, state, parameters, labels, replicate_indices)
    else:
        storage = _load_parameters_from_dict(fmt, state, parameters, labels, replicate_indices)

    for key, loaded_parameter in storage("parameters.all").walkitems():
        parameter, label = _get_parameter_value_label(parameters, labels, key, replicate_indices)

        parameter_value, parameter_error = parameter

        loaded_value = loaded_parameter.to_dict()["value"]
        loaded_error = loaded_parameter.to_dict()["sigma"]
        loaded_label = loaded_parameter.to_dict()["label"]

        assert parameter_value == loaded_value, "Loaded value is not equal to initial"
        if sigma_format == "sigma_absolute":
            assert parameter_error == loaded_error, (
                "Loaded absolute sigma" " is not equal to initial (absolute case)"
            )
        elif sigma_format == "sigma_relative":
            assert allclose(
                float64(abs(parameter_value) * parameter_error), loaded_error
            ), "Loaded sigma is not equal to initial (relative case)"
        else:
            assert allclose(
                float64(abs(parameter_value) * parameter_error * 1e-2), loaded_error
            ), "Loaded sigma is not equal to initial (percent case)"
        assert parameter_value == loaded_value, "Loaded value is not equal to initial"
        assert len(storage(f"parameters.{STATE_FORMAT_to_NS[(state, fmt)]}").keys()) > 0
