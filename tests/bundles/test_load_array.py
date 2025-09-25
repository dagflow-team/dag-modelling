import tempfile

import h5py
import numpy as np
import pytest

from dag_modelling.bundles.load_array import load_array


def _save_data(filename, object_name, data):
    _, extension = filename.split(".")
    if extension == "tsv":
        np.savetxt(filename, data)
    elif extension == "npz":
        np.savez(filename, **{object_name: data})
    elif extension == "hdf5":
        with h5py.File(filename, "w") as f:
            f.create_dataset(object_name, data=data)
    elif extension == "root":
        raise RuntimeError("It is not implemented")


# TODO: Add .root
@pytest.mark.parametrize("object_type", ["hdf5", "npz", "tsv"])
@pytest.mark.parametrize(
    "size,dtype", [((10, 10), "d"), ((12, 10), "f"), ((5,), "d"), ((1, 15), "f")]
)
def test_load_array(object_type, size, dtype):
    object_name = "matrix"
    output_ns = "array"
    output_name = "loaded_matrix"
    atol = np.finfo(dtype).resolution

    # NOTE
    # tsv data should be dumped in specific way
    # prefix_{object_name}.tsv -- {object_name} refers to key in .root/.hdf5/.npz files
    suffix = f"_{object_name}.{object_type}" if object_type == "tsv" else f".{object_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filename = f.name
        filename_prefix, _ = filename.split(suffix)
        filename_object = f"{filename_prefix}.{object_type}"
        generated_data = np.random.random(size=size)
        _save_data(filename, object_name, generated_data)

        storage = load_array(
            name=output_ns,
            filenames=filename_object,
            replicate_outputs=(output_name,),
            name_function={output_name: object_name},
            dtype=dtype,
        )

    loaded_array = storage[f"outputs.{output_ns}.{output_name}"]
    assert np.allclose(
        loaded_array.data, generated_data, atol=atol, rtol=0
    ), "Generated array is not equal to loaded"
