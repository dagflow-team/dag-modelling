from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from h5py import File
from numpy import ndarray, savez
from pandas import DataFrame

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Literal

    from numpy.typing import NDArray


def _save_records(
    filename: Path,
    *,
    dataframes: Mapping[str, DataFrame],
    records: Mapping[str, NDArray],
    tsv_kwargs: Mapping[str, Any] = {},
    pdhdf_kwargs: Mapping[str, Any] = {},
    tsv_include_parent_name: bool = False,
    tsv_allow_no_key: bool = False,
    tsv_mode: Literal["flat", "folder"] = "folder",
) -> None:
    Path(filename.parent).mkdir(parents=True, exist_ok=True)

    print_filename_global = True
    match filename.name.split("."):
        case (*_, "-"):
            for name, df in dataframes.items():
                print(name)
                print(df)
            print_filename_global = False
        case (*_, "npz"):
            savez(filename, **records)
        case (*_, "pd", "hdf5"):
            mode = "w"
            pdhdf_kwargs = dict(pdhdf_kwargs, index=False)
            for key, df in dataframes.items():
                df.to_hdf(filename, key=key, mode=mode, **pdhdf_kwargs)
                mode = "a"
        case (*_, "hdf5"):
            with File(filename, "w") as f:
                for key, record in records.items():
                    f.create_dataset(key, data=record)
        case (*_, "root"):
            from uproot import recreate

            file = recreate(filename)
            for key, record in records.items():
                file[key] = record
        case (*_, "tsv" | "txt"):
            name_no_key = len(dataframes) == 1 and tsv_allow_no_key
            print_filename_global = False
            match (tsv_mode, name_no_key):
                case _, True:
                    print_filename_global = True

                    def namefcn(_: str) -> Path | str:  # pyright: ignore [reportRedeclaration]
                        return filename

                case "flat", False:

                    def namefcn(key: str) -> Path | str:
                        return f"{filename.stem!s}_{key}{filename.suffix!s}"

                case "folder", False:
                    filename.mkdir(parents=True, exist_ok=True)
                    if tsv_include_parent_name:

                        def namefcn(key: str) -> Path | str:
                            return filename / f"{filename.stem!s}_{key}{filename.suffix!s}"

                    else:

                        def namefcn(key: str) -> Path | str:
                            return filename / f"{key}{filename.suffix!s}"

                case _:
                    raise ValueError(tsv_mode)

            tsv_kwargs = dict(tsv_kwargs)
            tsv_kwargs.setdefault("index", False)
            tsv_kwargs.setdefault("sep", "\t")

            for key, df in dataframes.items():
                koutput = namefcn(key)
                df.to_csv(koutput, **tsv_kwargs)
                if not print_filename_global:
                    print(f"Write {koutput}")
        case _:
            raise ValueError(f"Invalid file format: {filename}")

    if print_filename_global:
        print(f"Write {filename}")


def save_records(
    data: Mapping[str, NDArray | DataFrame],
    filenames: Path | str | Sequence[Path | str],
    to_records_kwargs: dict = {},
    **kwargs,
) -> None:
    records, dataframes = {}, {}
    for key, value in data.items():
        match value:
            case ndarray():
                records[key] = value
                dataframes[key] = DataFrame(value)
            case DataFrame():
                records[key] = value.to_records(**to_records_kwargs)
                dataframes[key] = value

    if not isinstance(filenames, Sequence):
        _save_records(Path(filenames), dataframes=dataframes, records=records, **kwargs)
        return

    for filename in filenames:
        _save_records(Path(filename), dataframes=dataframes, records=records, **kwargs)
