import tempfile

import numpy as np
import pytest
import uproot

from dag_modelling.bundles.load_hist import load_hist


def _save_data(filename, object_name, hist):
    _, extension = filename.split(".")
    if extension == "root":
        f = uproot.recreate(filename)
        f[object_name] = hist


# TODO: Add .hdf5, .npz, .tsv
@pytest.mark.parametrize("object_type", ["root"])
@pytest.mark.parametrize("entries,dtype", [(1, "d"), (100, "f")])
def test_load_hist(object_type, entries, dtype):
    object_name = "histogram"
    output_ns = "histogram"
    output_name = "loaded_histogram"
    atol = np.finfo(dtype).resolution

    # NOTE
    # tsv data should be dumped in specific way
    # prefix_{object_name}.tsv -- {object_name} refers to key in .root/.hdf5/.npz files
    suffix = f"_{object_name}.{object_type}" if object_type == "tsv" else f".{object_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filename = f.name
        filename_prefix, _ = filename.split(suffix)
        filename_object = filename_prefix + "." + object_type
        generated_data = np.random.random(size=entries)
        hist, bin_edges = np.histogram(generated_data)
        _save_data(filename, object_name, (hist, bin_edges))
        generated_data_dict = {"x": bin_edges, "y": hist}

        storage = load_hist(
            name=output_ns,
            filenames=filename_object,
            replicate_outputs=(output_name,),
            name_function={output_name: object_name},
            dtype=dtype,
        )

    for (axis, loaded_output_name), output in storage(f"outputs.{output_ns}").walkitems():
        assert loaded_output_name == output_name, "Initial output name is not equal to loaded"
        assert np.allclose(
            output.data, generated_data_dict[axis], atol=atol, rtol=0
        ), "Generated data is not equal to loaded"
