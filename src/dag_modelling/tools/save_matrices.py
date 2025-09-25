from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from h5py import File
from numpy import savetxt, savez_compressed

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Literal

    from numpy.typing import NDArray


def _save_matrices(
    filename: Path,
    *,
    matrices: Mapping[str, NDArray],
    tsv_kwargs: Mapping[str, Any] = {},
    tsv_include_parent_name: bool = False,
    tsv_allow_no_key: bool = False,
    tsv_mode: Literal["flat", "folder"] = "folder",
    root_reproducible: bool = False,
) -> None:
    Path(filename.parent).mkdir(parents=True, exist_ok=True)

    print_filename_global = True
    match filename.name.split("."):
        case (*_, "-"):
            for name, matrix in matrices.items():
                print(name)
                print(matrix)
            print_filename_global = False
        case (*_, "npz"):
            savez_compressed(filename, **matrices)
        case (*_, "hdf5"):
            with File(filename, "w") as f:
                for key, matrix in matrices.items():
                    f.create_dataset(key, data=matrix)
        case (*_, "root"):
            try:
                from ROOT import TFile, TMatrixD
            except ImportError as ex:
                raise RuntimeError("ROOT not found") from ex
            else:
                if root_reproducible:
                    filename_str = f"{filename!s}?reproducible={filename.name}"
                else:
                    filename_str = str(filename)
                file = TFile(filename_str, "recreate")
                for name, matrix in matrices.items():
                    Matrix = TMatrixD(matrix.shape[0], matrix.shape[1], matrix.ravel())
                    file.WriteTObject(Matrix, name, "overwrite")
                file.Close()
        case (*_, "tsv" | "txt") | (*_, "tsv" | "txt", "gz" | "bz2"):
            name_no_key = len(matrices) == 1 and tsv_allow_no_key
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
            tsv_kwargs.setdefault("delimiter", "\t")
            tsv_kwargs.setdefault("fmt", "%.17g")

            for key, matrix in matrices.items():
                koutput = namefcn(key)
                savetxt(koutput, matrix, **tsv_kwargs)
                if not print_filename_global:
                    print(f"Write {koutput}")
        case _:
            raise ValueError(f"Invalid file format: {filename}")

    if print_filename_global:
        print(f"Write {filename}")


def save_matrices(
    matrices: Mapping[str, NDArray],
    filenames: Path | str | Sequence[Path | str],
    **kwargs,
) -> None:
    if isinstance(filenames, (Path, str)):
        _save_matrices(Path(filenames), matrices=matrices, **kwargs)
        return

    for filename in filenames:
        _save_matrices(Path(filename), matrices=matrices, **kwargs)
