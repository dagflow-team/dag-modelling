from numpy import sqrt
from scipy.linalg import cholesky

from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.node import Node
from ...core.type_functions import check_node_has_inputs, check_inputs_are_matrices_or_diagonals, copy_from_inputs_to_outputs


class Cholesky(Node):
    """Compute the Cholesky decomposition of a matrix V=LL̃ᵀ
    1d input is considered to be a diagonal of square matrix"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "input_strategy",
            AddNewInputAddNewOutput(input_fmt="matrix", output_fmt="L"),
        )
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "V→L")

        self._functions_dict.update({"square": self._fcn_square, "diagonal": self._fcn_diagonal})

    def _fcn_square(self):
        """Compute Cholesky decomposition using `scipy.linalg.cholesky`
        NOTE: inplace computation (`overwrite_a=True`) works only for
        the F-based arrays. As soon as by default C-arrays are used,
        transposition produces an F-array (view). Transposition with
        `lower=False` produces a lower matrix in the end.
        """
        self.inputs.touch()

        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            outdata[:] = indata
            cholesky(outdata.T, overwrite_a=True, lower=False)  # produces L (!) inplace

    def _fcn_diagonal(self):
        """Compute "Cholesky" decomposition using of a diagonal of a square matrix.
        Elementwise sqrt is used.
        """
        self.inputs.touch()

        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            sqrt(indata, out=outdata)

    def _type_function(self) -> None:
        check_node_has_inputs(self)
        ndim = check_inputs_are_matrices_or_diagonals(self, slice(None), check_square=True)
        copy_from_inputs_to_outputs(self, slice(None), slice(None))

        if ndim == 2:
            self.function = self._functions_dict["square"]
            self.labels.mark = "V→L"
        else:
            self.function = self._functions_dict["diagonal"]
            self.labels.mark = "sqrt(Vᵢ)"
