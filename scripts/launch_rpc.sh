#!/bin/bash
function run() {
    device_id=$1
        echo "Starting rpc server on adb device: $device_id"
    preamble='
        spawn adb -s'
        body='shell
        expect "#"
    set cmd "cd /data/local/tmp/; LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --tracker=IP_ADDRESS:PORT --key=\"your_key_here\""
        send $cmd
        send "\r"
        interact
        '
        expect -c "$preamble $device_id $body"
}

while true
do
        run $1
        echo "RPC server died; restarting..." >&2
        sleep 20
done
