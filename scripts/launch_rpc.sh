#!/bin/bash

while true
do
   preamble='adb -s'
    body='shell cd /data/local/tmp/tvm_rpc; LD_LIBRARY_PATH=/data/local/tmp/tvm_rpc /data/local/tmp/tvm_rpc/tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091 --tracker=192.168.1.57:9190 --key=android
        '
        echo $preamble $1 $body
        $preamble $1 $body

    sleep 20
done
