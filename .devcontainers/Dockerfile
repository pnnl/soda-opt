FROM --platform=linux/amd64 agostini01/soda-bookworm:v19.00 as base
ARG DEBIAN_FRONTEND=noninteractive

FROM base as devcontainer
ARG DEBIAN_FRONTEND=noninteractive
# Add non-root user to the image.
ARG USERNAME="developer"
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Install a recent version of CMake not yet available via apt.
# Install Python kernel for use with Jupyter Notebooks.
# Install other dependencies from mlir. See: https://github.com/llvm/llvm-project/blob/release/19.x/mlir/python/requirements.txt
# TODO: if context path changes, this must be updated
COPY --chown=$USERNAME:$USERNAME requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set compilers to clang
ENV CC=clang
ENV CXX=clang++

ENV SHELL /bin/bash
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# Add mlir and llvm to python path
# ENV PYTHONPATH="/workspaces/mlir/python:/workspaces/llvm-project/llvm/bindings/python:${PYTHONPATH}"

# Allow container to stop. See: https://github.com/microsoft/vscode-remote-release/issues/3512#issuecomment-1267053890
# CMD while sleep 120 && ps aux | egrep 'vscode.*[b]ootstrap-fork.*extensionHost.*' > /dev/null; do :; done