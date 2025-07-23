import pytest
import tempfile
import h5py
import numpy as np
import pandas as pd
from dag_modelling.bundles.load_record import load_record


def _save_data(filename, object_name, data):
    _, extension = filename.split(".")
    if extension == "tsv":
        pd.DataFrame(data).to_csv(filename, sep="\t")
    elif extension == "npz":
        np.savez(filename, **{object_name: data})
    elif extension == "hdf5":
        with h5py.File(filename, "w") as f:
            f.create_dataset(object_name, data=data)
    elif extension == "root":
        raise Exception("It is not implemented")


# TODO: Add .root
@pytest.mark.parametrize("object_type", ["hdf5", "npz", "tsv"])
@pytest.mark.parametrize(
    "size,dtype",
    [
        ((2, 10), [("col0", "f"), ("col1", "f")]),
        ((3,), [("col0", "d"), ("col1", "f"), ("col2", "d")]),
        ((1, 15), [("col0", "f")])
    ]
)
def test_load_array(object_type, size, dtype):
    object_name = "matrix"
    output_ns = "record"
    output_name = "loaded_matrix"
    columns = [col[0] for col in dtype]

    # NOTE
    # tsv data should be dumped in specific way
    # prefix_{object_name}.tsv -- {object_name} refers to key in .root/.hdf5/.npz files
    suffix = f"_{object_name}.{object_type}" if object_type == "tsv" else f".{object_type}"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        filename = f.name
        filename_prefix, _ = filename.split(suffix)
        filename_object = filename_prefix + "." + object_type
        generated_data = np.random.random(size=size)
        generated_data = np.array(generated_data, dtype=dtype).reshape(-1)
        _save_data(filename, object_name, generated_data)

        storage = load_record(
            name=output_ns,
            filenames=filename_object,
            replicate_outputs=(output_name,),
            name_function={output_name: object_name},
            columns=columns,
        )


    for (column, loaded_output_name), output in storage(f"outputs.{output_ns}").walkitems():
        atol = np.finfo(generated_data[column].dtype).resolution
        assert loaded_output_name == output_name, "Initial output name is not equal to loaded"
        assert np.allclose(output.data, generated_data[column], atol=atol, rtol=0), "Generated data is not equal to loaded"
