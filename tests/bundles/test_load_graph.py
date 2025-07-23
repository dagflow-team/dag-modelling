import tempfile

import h5py
import numpy as np
import pytest

from dag_modelling.bundles.load_graph import load_graph


def _save_data(filename, object_name, data):
    _, extension = filename.split(".")
    if extension == "tsv":
        with open(filename, "w") as f:
            f.write("x\ty\n")
            f.writelines([f"{x[0]:1.32e}" + "\t" + f"{x[1]:1.32e}\n" for x in data])
    elif extension == "npz":
        np.savez(filename, **{object_name: data})
    elif extension == "hdf5":
        with h5py.File(filename, "w") as f:
            f.create_dataset(object_name, data=data)
    elif extension == "root":
        raise Exception("It is not implemented")


# TODO: Add .root
@pytest.mark.parametrize("object_type", ["tsv", "hdf5", "npz", "tsv"])
@pytest.mark.parametrize("n_points", [1, 2, 10])
@pytest.mark.parametrize("x_axis_parameters,dtype", [((-10, 10), "d"), ((10, 20), "f")])
def test_load_graph(object_type, n_points, x_axis_parameters, dtype):
    object_name = "spectrum"
    output_ns = "graph"
    output_name = "loaded_spectrum"
    x_start, x_stop = x_axis_parameters
    column_x, column_y = ("x", "y")
    atol = np.finfo(dtype).resolution

    # NOTE
    # tsv data should be dumped in specific way
    # prefix_{object_name}.tsv -- {object_name} refers to key in .root/.hdf5/.npz files
    suffix = f"_{object_name}.{object_type}" if object_type == "tsv" else f".{object_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filename = f.name
        filename_prefix, _ = filename.split(suffix)
        filename_object = filename_prefix + "." + object_type
        generated_x = np.linspace(x_start, x_stop, n_points)
        generated_y = np.random.random(size=n_points)
        generated_data = np.array(
            list(zip(generated_x, generated_y)),
            dtype=[(column_x, dtype), (column_y, dtype)],
        )
        _save_data(filename, object_name, generated_data)

        storage = load_graph(
            name=output_ns,
            filenames=filename_object,
            replicate_outputs=(output_name,),
            name_function={output_name: object_name},
            x=column_x,
            y=column_y,
            dtype=dtype,
        )

    for (column, loaded_output_name), output in storage(f"outputs.{output_ns}").walkitems():
        assert loaded_output_name == output_name, "Initial output name is not equal to loaded"
        assert np.allclose(
            output.data, generated_data[column], atol=atol, rtol=0
        ), "Generated data is not equal to loaded"
