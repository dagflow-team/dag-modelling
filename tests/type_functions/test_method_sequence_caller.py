from dag_modelling.core.type_functions.tools_for_type_functions import MethodSequenceCaller


def test_00():
    caller = MethodSequenceCaller()
    x = []
    caller.add(lambda x, y: x.append(y))
    caller.add(lambda x, _: x.pop())
    caller(x, 1)
    assert not x


def test_01():
    caller = MethodSequenceCaller()
    x = []
    caller.add(lambda x, y: x.append(y))
    caller.add(lambda x, y: x.append(y + 1))
    caller.add(lambda x, y: x.append(y * 11))
    caller(x, 1)
    assert x == [1, 2, 11]
