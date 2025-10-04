# Synthetic graph

Attempt to create cython kernel for running dag-modelling.

## How to run tests

1. Compile cython kernel
```bash
cd synthetic_graph/graph_cython
python setup.py build_ext --inplace
```
2. Go back to the root of project and update `PYTHONPATH` variable
```bash
export PYTHONPATH=$PYTHONPATH:$PWD/
export PYTHONPATH=$PYTHONPATH:$PWD/synthetic_graph/graph_cython/build/lib
```
3. Run simple test for native dag-modelling
```bash
python synthetic_graph/tests/graph-df.py
```
4. Run simple test for cython kernel of dag-modelling
```bash
python synthetic_graph/tests/graph.py
```

## Docker solution

You can find `Dockerfile` and `.dockerignore` in `synthetic_graph/misc`. Copy them to the root of project and build image for running dag-modelling in container environment.

1. Copy `Dockerfile` and `.dockerignore` to the root of project:
```bash
cp synthetic_graph/misc/Dockerfile ./
cp synthetic_graph/misc/.dockerignore ./
```
2. Build docker image
```bash
docker build -t dag-modelling:test .
```
3. Run docker container and install dag-modelling within it
```bash
docker run -v .:/app -it dag-modelling:tests bash
pip install -e ".[test]"
```
4. Check that pytest works well
```bash
pytest tests/
```
