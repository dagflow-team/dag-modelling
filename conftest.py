from os import chdir, environ, getcwd, listdir, mkdir
from os.path import isdir

from pytest import fixture


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.

    Automatic change path to the `dag-modelling/tests` and create `tests/output` dir
    """
    while path := getcwd():
        if (lastdir := path.split("/")[-1]) == "tests":
            break
        elif ".git" in listdir(path):
            chdir("./tests")
            break
        else:
            chdir("..")
    if not isdir("output"):
        mkdir("output")


def pytest_addoption(parser):
    parser.addoption(
        "--debug-graph",
        action="store_true",
        default=False,
        help="set debug=True for all the graphs in tests",
    )
    parser.addoption(
        "--include-long-time-tests", action="store_true", default=False, help="include long-time tests"
    )


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.include_long_time_tests
    if "--include-long-time-tests" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("--include-long-time-tests", [option_value])


########################################
# Create global variables (fixtures) for a pytest run
########################################
@fixture(scope="session")
def debug_graph(request):
    return request.config.option.debug_graph


@fixture()
def testname():
    """Returns corrected full name of a test"""
    name = environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    name = name.replace("[", "_").replace("]", "")
    return name
