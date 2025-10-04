FROM 'python:3.11.7-bullseye'

ARG USER=user
ARG APPDIR=/app
ENV VIRTUAL_ENV=/venv

RUN apt update && apt install -y libgraphviz-dev graphviz --no-install-recommends
WORKDIR /app

RUN python3 -m venv ${VIRTUAL_ENV} \
  && . ${VIRTUAL_ENV}/bin/activate \
  && pip install --no-cache-dir --upgrade wheel setuptools pip \
  && pip install --no-cache-dir cython

RUN apt autoremove -y \
  && apt autoclean -y \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /tmp/*

RUN groupadd --gid 1000 $USER \
  && useradd --uid 1000 --gid $USER $USER \
  && chown -R $USER:$USER /app

RUN mkdir /home/${USER} \
  && chown -R ${USER}:${USER} /home/${USER} \
  && chown -R ${USER}:${USER} ${APPDIR} \
  && chown -R ${USER}:${USER} ${VIRTUAL_ENV}

USER $USER
ENV PYTHONPATH=${VIRTUAL_ENV}/bin/ PATH=${VIRTUAL_ENV}/bin/:$PATH
