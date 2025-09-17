from os import environ, makedirs

from pytest import fixture


def pytest_addoption(parser):
    parser.addoption(
        "--debug-graph",
        action="store_true",
        default=False,
        help="set debug=True for all the graphs in tests",
    )
    parser.addoption(
        "--include-long-time-tests",
        action="store_true",
        default=False,
        help="include long-time tests",
    )
    parser.addoption(
        "--output-path",
        default="output/tests",
        help="choose the location of output materials",
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.include_long_time_tests
    if "--include-long-time-tests" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("--include-long-time-tests", [option_value])


@fixture(scope="session")
def output_path(request):
    loc = request.config.option.output_path
    makedirs(loc, exist_ok=True)
    return loc


@fixture(scope="session")
def debug_graph(request):
    return request.config.option.debug_graph


@fixture()
def test_name():
    """Returns corrected full name of a test."""
    name = environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    name = name.replace("[", "_").replace("]", "")
    return name
