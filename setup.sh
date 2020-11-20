#!/bin/bash

pathmunge () {    
    
    ENV_VAR_NAME=$(echo $1 | tr '[:lower:]' '[:upper:]')
    
    eval ENV_VAR=\$$ENV_VAR_NAME    
    if ! echo $ENV_VAR | /bin/egrep -q "(^|:)$2($|:)" ; then
        if [ "$3" = "after" ] ; then
	    ENV_VAR=$ENV_VAR:$2
        else
            ENV_VAR=$2:$ENV_VAR
        fi
    fi
    eval ${ENV_VAR_NAME}=$ENV_VAR
}


BEGEPATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
pathmunge PYTHONPATH $BEGEPATH/src

export PYTHONPATH=$PYTHONPATH
#export PATH=$PATH



