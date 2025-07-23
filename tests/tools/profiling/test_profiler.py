from collections import Counter

from pytest import raises

from dag_modelling.tools.profiling.profiler import Profiler


def test_init_g0(monkeypatch, graph_0):
    monkeypatch.setattr(Profiler, "__abstractmethods__", set())
    _, nodes = graph_0
    a0, a1, a2, a3, _, p1, s0, s1, s2, s3, l_matrix, mdvdt = nodes

    target_nodes = [p1, s1, s2]
    profiling = Profiler(target_nodes)
    assert profiling._target_nodes == target_nodes
    assert profiling._sources == profiling._sinks == ()

    sources, sinks = [a2, a3], [s3]
    target_nodes = [a2, a3, s0, p1, s1, s2, s3]
    profiling = Profiler(sources=sources, sinks=sinks)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)
    assert profiling._sources == sources
    assert profiling._sinks == sinks

    sources, sinks = [a0, a1, a2, a3, l_matrix], [s3, mdvdt]
    target_nodes = nodes
    profiling = Profiler(sources=sources, sinks=sinks)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)

    sources, sinks = [a2, a3], [l_matrix]
    with raises(ValueError) as excinfo:
        Profiler(sources=sources, sinks=sinks)
    assert "unreachable" in str(excinfo.value)

    with raises(ValueError) as excinfo:
        Profiler()
    assert "You shoud provide profiler with `target_nodes`" in str(excinfo.value)


def test_init_g1(monkeypatch, graph_1):
    monkeypatch.setattr(Profiler, "__abstractmethods__", set())
    _, nodes = graph_1
    a0, a1, a2, a3, a4, s1, s2, p1, p2 = nodes

    sources, sinks = [a4, s1], [p2]
    target_nodes = [a4, s1, p1, p2]
    profiling = Profiler(sources=sources, sinks=sinks)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)
    assert profiling._sources == sources
    assert profiling._sinks == sinks

    sources, sinks = [a0, a1, a2, a3, a4], [p2]
    target_nodes = nodes
    profiling = Profiler(sources=sources, sinks=sinks)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)

    sources, sinks = [a0, a2], [p1]
    target_nodes = [a0, a2, s1, p1]
    profiling = Profiler(sources=sources, sinks=sinks)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)

    sources, sinks = [a0, a1], [s2]
    with raises(ValueError) as excinfo:
        Profiler(sources=sources, sinks=sinks)
    assert "unreachable" in str(excinfo.value)
