from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .output import Output

ShapeLike = tuple[int, ...]
LayoutLike = Literal["dense", "sparse-csr"]
EdgesLike = tuple[Output, ...]
MeshesLike = tuple[Output, ...]
