from __future__ import annotations

from contextlib import suppress
from os import listdir
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import concatenate, double, dtype, frombuffer, linspace, ndarray

from nested_mapping.tools.map import make_reorder_function
from nested_mapping.typing import properkey

from ..tools.logger import INFO1, INFO2, INFO3, logger

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from typing import Any

    import ROOT
    from numpy.typing import NDArray

    from nested_mapping.typing import KeyLike, TupleKey

file_readers = {}

_log_float_format = ".3g"


class HistGetter:
    __slots__ = ()

    def __getitem__(self, names: tuple[str | Path, str]) -> tuple[NDArray, NDArray]:
        file_name, object_name = names
        fr = FileReader[file_name]
        return fr.get_hist(object_name)


_HistGetter = HistGetter()


class GraphGetter:
    __slots__ = ()

    def __getitem__(self, names: tuple[str | Path, str]) -> tuple[NDArray, NDArray]:
        file_name, object_name = names
        fr = FileReader[file_name]
        return fr.get_graph(object_name)


_GraphGetter = GraphGetter()


class ArrayGetter:
    __slots__ = ()

    def __getitem__(self, names: tuple[str | Path, str]) -> NDArray:
        file_name, object_name = names
        fr = FileReader[file_name]
        return fr.get_array(object_name)


_ArrayGetter = ArrayGetter()


class RecordGetter:
    __slots__ = ()

    def __getitem__(self, names: tuple[str | Path, str]) -> NDArray | dict[str, NDArray]:
        file_name, object_name = names
        fr = FileReader[file_name]
        return fr.get_record(object_name)


_RecordGetter = RecordGetter()


class FileReaderMeta(type):
    """Metaclass for `FileReader` class, implementing `FileReader[file_name]`
    method."""

    _opened_files: dict[str, FileReader] = {}
    _last_used_file: str = ""

    def __init__(self, name: str, parents: tuple, args: dict) -> None:
        """Register the file reader based on the `_extension`"""
        super().__init__(name, parents, args)
        try:
            ext = args["_extension"]
        except KeyError as e:
            raise e
        else:
            file_readers[ext] = self

    def __getitem__(self, file_name: str | Path) -> FileReader:
        file_name_str = file_name if isinstance(file_name, str) else str(file_name)
        try:
            ret = self._opened_files[file_name_str]
            action = "Use" if file_name_str != self._last_used_file else None
        except KeyError:
            ret = FileReader.open(file_name)
            action = "Read"

        if action:
            logger.log(INFO1, f"{action}: {file_name_str}")

        self._opened_files[file_name_str] = ret
        self._last_used_file = file_name_str

        return ret

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args, **kwargs) -> None:
        self.release_files()

    def release_files(self) -> None:
        for k, v in list(self._opened_files.items()):
            v._close()
            del self._opened_files[k]

            logger.log(INFO3, f"Close: {v._file_name!s}")

    @property
    def array(self) -> ArrayGetter:
        return _ArrayGetter

    @property
    def graph(self) -> GraphGetter:
        return _GraphGetter

    @property
    def hist(self) -> HistGetter:
        return _HistGetter

    @property
    def record(self) -> RecordGetter:
        return _RecordGetter


class FileReader(metaclass=FileReaderMeta):
    _extension: str = ""
    _file: Any = None
    _file_name: Path = Path("")
    _opened_files: dict[str, FileReader] = FileReaderMeta._opened_files
    _read_objects: dict[str, Any]

    def __init__(self, file_name: str | Path):
        self._file_name = Path(file_name)
        self._read_objects = {}

    @classmethod
    def open(cls, file_name: str | Path) -> FileReader:
        file_path = file_name if isinstance(file_name, Path) else Path(file_name)
        ext = file_path.suffix

        try:
            cls = file_readers[ext]
        except KeyError as e:
            raise ValueError(
                f"Do not know how to load ext {ext}. Available file_readers:"
                f" {', '.join(file_readers)}"
            ) from e

        try:
            return cls(file_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Can not open file {file_name!s} (loader {ext})") from e

    def _close(self) -> None:
        self._read_objects = {}

    def keys(self) -> tuple[str, ...]:
        raise RuntimeError("not implemented method")

    def _get_object_impl(self, object_name: str, **kwargs) -> Any:
        raise RuntimeError("not implemented method")

    def _get_object(self, object_name: str, **kwargs) -> Any:
        object_name = object_name.replace(".", "_")
        with suppress(KeyError):
            return self._read_objects[object_name]

        try:
            self._read_objects[object_name] = (
                object := self._get_object_impl(object_name, **kwargs)
            )
            return object
        except KeyError as e:
            raise KeyError(f"Can not read {object_name} from {self._file_name!s}") from e

    def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        x, y = self._get_graph(object_name)
        logger.log(
            INFO2,
            f"graph {object_name} ({len(y)}): x"
            f" {x[0]:{_log_float_format}}→{x[-1]:{_log_float_format}},"
            f" ymin={y.min():{_log_float_format}}, ymax={y.max():{_log_float_format}}",
        )
        return x, y

    def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        raise RuntimeError("not implemented method")

    def get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        x, y = self._get_hist(object_name)
        logger.log(
            INFO2,
            f"hist {object_name} ({len(y)}): x"
            f" {x[0]:{_log_float_format}}→{x[-1]:{_log_float_format}},"
            f" min={y.min():{_log_float_format}}, max={y.max():{_log_float_format}},"
            f" Σh={y.sum():{_log_float_format}}",
        )
        return x, y

    def _get_array(self, object_name: str) -> NDArray:
        raise RuntimeError("not implemented method")

    def get_array(self, object_name: str) -> NDArray:
        a = self._get_array(object_name)
        logger.log(
            INFO2,
            f"array {object_name} {'x'.join(map(str,a.shape))}: min={a.min():{_log_float_format}},"
            f" max={a.max():{_log_float_format}}, Σ={a.sum():{_log_float_format}}",
        )
        return a

    def _get_record(self, object_name: str) -> NDArray | dict[str, NDArray]:
        raise RuntimeError("not implemented method")

    def get_record(self, object_name: str) -> NDArray | dict[str, NDArray]:
        rec = self._get_record(object_name)

        match rec:
            case ndarray():
                nrows = rec.shape[0]
                columns = ", ".join(rec.dtype.names) if rec.dtype.names else "???"
            case dict():
                nrows = next(iter(rec.values())).shape[0]
                columns = ", ".join(rec.keys())
            case _:
                nrows = -1
                columns = "???"

        logger.log(
            INFO2,
            f"record {object_name} ({nrows}):" f" {columns}",
        )
        return rec


class FileReaderArray(FileReader):
    _extension: str = ""

    def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        return self._get_xy(object_name)

    def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        emin, emax, h = self._get_emin_emax_h(object_name)
        if not (emin[1:] == emax[:-1]).all():
            raise ValueError("Inconsistent histogram edges")
        edges = concatenate((emin, emax[-1:]))
        return edges, h

    def _get_array(self, object_name: str) -> NDArray:
        return self._get_object(object_name)

    def _get_record(self, object_name: str) -> NDArray:
        return self._get_object(object_name)

    def _get_xy(self, object_name: str) -> tuple[NDArray, NDArray]:
        data = self._get_object(object_name)
        cols = data.dtype.names
        return data[cols[0]], data[cols[1]]

    def _get_emin_emax_h(self, object_name: str) -> tuple[NDArray, NDArray, NDArray]:
        data = self._get_object(object_name)
        cols = data.dtype.names
        return data[cols[0]], data[cols[1]], data[cols[2]]


class FileReaderNPZ(FileReaderArray):
    _extension: str = ".npz"

    def __init__(self, file_name: str | Path) -> None:
        super().__init__(file_name)
        from numpy import load

        self._file = load(self._file_name, allow_pickle=True)

    def _close(self) -> None:
        super()._close()
        del self._file

    def _get_object_impl(self, object_name: str, **kwargs) -> Any:
        assert not kwargs
        return self._file[object_name]

    def keys(self) -> tuple[str, ...]:
        return tuple(self._file.keys())


class FileReaderHDF5(FileReaderArray):
    _extension: str = ".hdf5"

    def __init__(self, file_name: str | Path) -> None:
        super().__init__(file_name)
        from h5py import File

        self._file = File(self._file_name, "r")

    def _close(self) -> None:
        super()._close()
        self._file.close()

    def _get_object_impl(self, object_name: str, **kwargs) -> Any:
        assert not kwargs
        return self._file[object_name]

    def _get_array(self, object_name: str) -> NDArray:
        ret = self._get_object(object_name)
        return ret[:]

    def keys(self) -> tuple[str, ...]:
        return tuple(self._file.keys())


class FileReaderTSV(FileReaderArray):
    _extension: str = ".tsv"

    def __init__(self, file_name: str | Path) -> None:
        super().__init__(file_name)

    def _get_filenames(self, object_name: str) -> tuple[str, ...]:
        uncompressed = (
            str(self._file_name / f"{object_name}{self._extension}"),
            f"{self._file_name.parent/self._file_name.stem}_{object_name}{self._extension}",
            str(self._file_name / f"{self._file_name.stem}_{object_name}{self._extension}"),
        )

        return uncompressed + tuple(f"{fname}.bz2" for fname in uncompressed)

    def _get_object_impl(self, object_name: str, return_record: bool = True) -> Any:
        filenames = self._get_filenames(object_name)

        if return_record:
            from pandas import read_table

            for filename in filenames:
                with suppress(FileNotFoundError):
                    df = read_table(filename, comment="#", sep=None, engine="python")
                    logger.log(INFO1, f"Read: {filename}")
                    return df.to_records(index=False)
        else:
            from numpy import loadtxt

            for filename in filenames:
                with suppress(FileNotFoundError):
                    ret = loadtxt(filename)
                    logger.log(INFO1, f"Read: {filename}")
                    return ret

        raise FileNotFoundError(", ".join(map(str, filenames)))

    def _get_array(self, object_name: str) -> NDArray:
        return self._get_object(object_name, return_record=False)

    def keys(self) -> tuple[str, ...]:
        return tuple(file for file in listdir(self._file_name) if file.endswith(self._extension))


class FileReaderROOTUpROOT(FileReader):
    _extension: str = ".root"

    def __init__(self, file_name: str | Path) -> None:
        super().__init__(file_name)
        from uproot import open

        self._file = open(file_name)

    def _close(self) -> None:
        super()._close()
        self._file.close()

    def _get_object_impl(self, object_name: str, **kwargs) -> Any:
        assert not kwargs
        return self._file[object_name]

    def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
        obj = self._get_object(object_name)
        y, x = obj.to_numpy()
        return x, y

    def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
        obj = self._get_object(object_name)
        y, x = obj.to_numpy()
        return x[:-1], y

    def _get_array(self, object_name: str) -> NDArray:
        obj = self._get_object(object_name)
        y, _ = obj.to_numpy()
        return y

    def _get_record(self, object_name: str) -> dict[str, NDArray]:
        tree = self._get_object(object_name)
        return {key: tree[key].array().to_numpy().copy() for key in tree.keys()}

    def keys(self) -> tuple[str, ...]:
        return tuple(key.split(";", 1)[0] for key in self._file.GetListOfKeys())


with suppress(ImportError):
    import ROOT

    class FileReaderROOTROOT(FileReader):
        _extension: str = ".root"
        _reader_uproot: FileReaderROOTUpROOT | None = None

        def __init__(self, file_name: str | Path) -> None:
            super().__init__(file_name)
            from ROOT import TFile

            self._file = TFile(file_name)
            if self._file.IsZombie():
                raise FileNotFoundError(file_name)

        @property
        def reader_uproot(self) -> FileReaderROOTUpROOT:
            if self._reader_uproot is None:
                self._reader_uproot = FileReaderROOTUpROOT(self._file_name)

            return self._reader_uproot

        def _close(self) -> None:
            super()._close()
            self._file.Close()

            if self._reader_uproot is not None:
                self._reader_uproot._close()

        def _get_object_impl(self, object_name: str, **kwargs) -> Any:
            assert not kwargs
            ret = self._file.Get(object_name)
            if not ret:
                raise KeyError(object_name)
            return ret

        def _get_hist(self, object_name: str) -> tuple[NDArray, NDArray]:
            import ROOT

            obj = self._get_object(object_name)
            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_bin_edges(obj.GetXaxis()), _get_buffer_hist1(obj, flows=False)

            raise ValueError(f"Do not know ho to convert {obj} to hist")

        def _get_graph(self, object_name: str) -> tuple[NDArray, NDArray]:
            import ROOT

            obj = self._get_object(object_name)
            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_bin_left_edges(obj.GetXaxis()), _get_buffer_hist1(obj, flows=False)
            if isinstance(obj, ROOT.TGraph):
                return _get_buffers_graph(obj)

            raise ValueError(f"Do not know ho to convert {obj} to graph")

        def _get_array(self, object_name: str) -> NDArray:
            import ROOT

            obj = self._get_object(object_name)

            if isinstance(obj, ROOT.TH1) and obj.GetDimension() == 1:
                return _get_buffer_hist1(obj)
            if isinstance(obj, ROOT.TH2) and obj.GetDimension() == 2:
                return _get_buffer_hist2(obj)
            if isinstance(obj, (ROOT.TMatrixD, ROOT.TMatrixF)):
                return _get_buffer_matrix(obj)

            raise ValueError(f"Do not know ho to convert {obj} to array")

        def _get_record(self, object_name: str) -> dict[str, NDArray]:
            return self.reader_uproot.get_record(object_name)

        def keys(self) -> tuple[str, ...]:
            return tuple(key.GetName().split(";", 1)[0] for key in self._file.GetListOfKeys())


def iterate_filenames(
    filenames: Sequence[str | Path], keys: Sequence[KeyLike]
) -> Generator[tuple[TupleKey, str | Path], None, None]:
    for keylike in keys:
        key = properkey(keylike)
        for afilename in filenames:
            filename = str(afilename)
            if "{" in filename:
                ffilename = filename.format(*key)
                yield key, ffilename
                break
            elif all(map(filename.__contains__, key)):
                yield key, afilename
                break
        else:
            raise RuntimeError(f"Could not find a file for key {'.'.join(key)}")


def iterate_filenames_and_objectnames(
    filenames: Sequence[str | Path],
    filename_keys: Sequence[KeyLike],
    keys: Sequence[KeyLike],
    *,
    skip: Sequence[set[str]] | None = None,
    key_order: Sequence[int] | None = None,
) -> Generator[tuple[TupleKey, str | Path, TupleKey, TupleKey], None, None]:
    reorder_key = make_reorder_function(key_order)
    for filekey, filename in iterate_filenames(filenames, filename_keys):
        for key in keys:
            key = properkey(key)
            fullkey = filekey + key
            if skip is not None and any(skipkey.issubset(fullkey) for skipkey in skip):
                continue
            fullkey = reorder_key(fullkey)
            yield filekey, filename, key, fullkey


def _get_buffer_hist1(h: ROOT.TH1, flows: bool = False) -> NDArray:
    """Return TH1* histogram data buffer if flows=False, exclude underflow and
    overflow."""
    buf = h.GetArray()
    buf = frombuffer(buf, dtype(buf.typecode), h.GetNbinsX() + 2)
    if not flows:
        buf = buf[1:-1]

    return buf.copy()


def _get_buffer_hist2(h, flows=False):
    """Return histogram data buffer
    if flows=False, exclude underflow and overflow
    NOTE: buf[biny][binx] is the right access signature
    """
    nx, ny = h.GetNbinsX(), h.GetNbinsY()
    buf = h.GetArray()
    res = frombuffer(buf, dtype(buf.typecode), (nx + 2) * (ny + 2)).reshape((ny + 2, nx + 2))
    if not flows:
        res = res[1 : ny + 1, 1 : nx + 1]

    return res.copy()


def _get_bin_edges(ax: ROOT.TAxis) -> NDArray:
    """Get the array with bin edges."""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n > 0:
        lims = frombuffer(xbins.GetArray(), double, n)
        return lims.copy()
    return linspace(ax.GetXmin(), ax.GetXmax(), ax.GetNbins() + 1)


def _get_bin_left_edges(ax: ROOT.TAxis) -> NDArray:
    """Get the array with bin left edges."""
    xbins = ax.GetXbins()
    n = xbins.GetSize()
    if n > 0:
        lims = frombuffer(xbins.GetArray(), double, n)
        return lims[:-1].copy()
    return linspace(ax.GetXmin(), ax.GetXmax(), ax.GetNbins() + 1)[:-1]


def _get_buffers_graph(g: ROOT.TGraph) -> tuple[NDArray, NDArray]:
    """Get TGraph x and y buffers."""
    npoints = g.GetN()
    if npoints == 0:
        raise RuntimeError("Got graph with 0 points")

    return (
        frombuffer(g.GetX(), dtype=double, count=npoints).copy(),
        frombuffer(g.GetY(), dtype=double, count=npoints).copy(),
    )


def _get_buffer_matrix(m):
    """Get TMatrix buffer."""
    cbuf = m.GetMatrixArray()
    res = frombuffer(cbuf, dtype(cbuf.typecode), m.GetNoElements()).reshape(
        m.GetNrows(), m.GetNcols()
    )
    return res.astype(double).copy()
