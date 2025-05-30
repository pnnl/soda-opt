#!/bin/bash

set -e
set -o pipefail

# Check if docker is available or if the needed binaries are available
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source $SCRIPT_DIR/check_docker.sh

BAMBUDIR=output/$1
pushd $BAMBUDIR

# F_GROUP_NAME=`stat -c "%G" $(pwd)`
# F_GROUP_ID=`cut -d: -f3 < <(getent group $F_GROUP_NAME)`
# docker run -u `id -u`:$F_GROUP_ID -v $PWD:/Panda-flow -it --rm panda_openroad:latest 

# docker run -u $(id -u):$(id -g) -v $PWD:/Panda-flow -v $PWD:/working_dir --rm agostini01/panda_openroad:latest \


$DOCKER_RUN \
/bin/bash ./synthesize_Synthesis_main_kernel.sh

popd