# Build from the dist/ directory, after choosing an approved wheel.
# Preserve the base image's original user via NOTEBOOK_USER (0 if unset).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG NOTEBOOK_USER
USER root
COPY kjob-0.5.0-py3-none-any.whl /tmp/kjob-0.5.0-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir /tmp/kjob-0.5.0-py3-none-any.whl && \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps /tmp/kjob-0.5.0-py3-none-any.whl && \
    rm /tmp/kjob-0.5.0-py3-none-any.whl
USER ${NOTEBOOK_USER}
