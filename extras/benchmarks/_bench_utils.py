"""Shared print helpers for the extras/benchmarks/ scripts."""

import io
import pstats


def banner(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_cprofile_stats(profiler, sort_by: str = "cumulative", n: int = 20) -> pstats.Stats:
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_by)
    stats.print_stats(n)
    print(stream.getvalue())
    return stats
