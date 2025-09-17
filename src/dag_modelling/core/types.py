from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .output import Output

ShapeLike = tuple[int, ...]
EdgesLike = tuple[Output, ...]
MeshesLike = tuple[Output, ...]
