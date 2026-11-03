import tempfile

import numpy as np
import pytest
import uproot

from numpy.typing import NDArray

from dag_modelling.bundles.load_graph import load_graph


def _create_hist_file(filename, suffix, object_name, object_type, entries) -> tuple[str, tuple[NDArray, NDArray]]:
    filename_prefix, _ = filename.split(suffix)
    filename_object = filename_prefix + "." + object_type
    generated_data = np.random.random(size=entries)
    hist, bin_edges = np.histogram(generated_data)
    _save_hist(filename, object_name, (hist, bin_edges))
    return filename_object, (hist, bin_edges)


def _save_hist(filename, object_name, hist):
    _, extension = filename.split(".")
    if extension == "root":
        f = uproot.recreate(filename)
        f[object_name] = hist


# TODO: Add .hdf5, .npz, .tsv
@pytest.mark.parametrize("object_type", ["root"])
@pytest.mark.parametrize(
    "entries,dtype,strategy", [(1, "d", "left"), (100, "f", "center"), (100, "d", "right")]
)
def test_load_hist_as_graph(object_type, entries, dtype, strategy):
    object_name = "histogram"
    output_ns = "histogram"
    output_name = "loaded_histogram"
    atol = np.finfo(dtype).resolution

    # NOTE: tsv data should be dumped in specific way
    # prefix_{object_name}.tsv -- {object_name} refers to key in .root/.hdf5/.npz files
    suffix = f"_{object_name}.{object_type}" if object_type == "tsv" else f".{object_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filename_object, (hist, bin_edges) = _create_hist_file(f.name, suffix, object_name, object_type, entries)
        x_points = (
            bin_edges[:-1]
            if strategy == "left"
            else bin_edges[1:] if strategy == "right" else (bin_edges[:-1] + bin_edges[1:]) / 2.0
        )
        generated_data_dict = {"x": x_points, "y": hist}

        storage = load_graph(
            name=output_ns,
            filenames=filename_object,
            replicate_outputs=(output_name,),
            name_function={output_name: object_name},
            dtype=dtype,
            hist_edges_strategy=strategy,
        )

    for (axis, loaded_output_name), output in storage(f"outputs.{output_ns}").walkitems():
        assert loaded_output_name == output_name, "Initial output name is not equal to loaded"
        assert np.allclose(
            output.data, generated_data_dict[axis], atol=atol, rtol=0
        ), "Generated data is not equal to loaded"


# TODO: Add .hdf5, .npz, .tsv
@pytest.mark.parametrize("object_type", ["root"])
@pytest.mark.parametrize(
    "entries,dtype",
    [
        (1, "d"),
    ],
)
def test_load_hist_as_graph_exceptions(object_type, entries, dtype):
    object_name = "histogram"
    output_ns = "histogram"
    output_name = "loaded_histogram"
    atol = np.finfo(dtype).resolution

    # NOTE: tsv data should be dumped in specific way
    # prefix_{object_name}.tsv -- {object_name} refers to key in .root/.hdf5/.npz files
    suffix = f"_{object_name}.{object_type}" if object_type == "tsv" else f".{object_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filename_object, (hist, bin_edges) = _create_hist_file(f.name, suffix, object_name, object_type, entries)

        with pytest.raises(RuntimeError):
            storage = load_graph(
                name=output_ns,
                filenames=filename_object,
                replicate_outputs=(output_name,),
                name_function={output_name: object_name},
                dtype=dtype,
            )


    # TODO: Add test for graph as hist
