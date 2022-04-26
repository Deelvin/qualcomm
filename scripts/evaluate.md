# Setup Android Environment
- Download Android Studio https://developer.android.com/studio#downloads
- Install Android Studio https://developer.android.com/studio/install#linux
- Launch Android Studio `/usr/local/android-studio/bin/studio.sh`
- Click `Customize`
- Click `All settings...`
- Select `Appearance & Behavior`
- Select `System settings`
- Select `Android SDK`
- Change if needed - Android SDK Location: `/home/avoronov/Android/Sdk`
- Click `SDK Tools`
- Select `Show Package Details`
- Select `SDK Tools -> Show Package Details`
- Select `NDK (Side by side) -> 21.4.7075529`
- Click `Apply` and wait for the download to complete.


# Build qualcomm repository
###### Get the repository
```
git clone https://github.com/Deelvin/qualcomm.git
cd qualcomm
git submodule init
git submodule update
```


###### Build host part
```
mkdir build_host
cp tvm/cmake/config.cmake build_host

echo 'set(USE_LLVM ON)' >> build_host/config.cmake
# or provide a direct path to llvm-config
echo 'set(USE_LLVM /usr/bin/llvm-config)' >> build_host/config.cmake

cmake -DCMAKE_BUILD_TYPE=Release ../tvm
make -j16
```


###### Build android part
```
mkdir build_android
cp tvm/cmake/config.cmake build_android

echo 'set(USE_HEXAGON_SDK OFF)' >> build_android/config.cmake
echo 'set(USE_CPP_RPC ON)' >> build_android/config.cmake
# optional if you wanted to specify a custom OpenCL
echo 'set(USE_OPENCL /home/avoronov/AdrenoOpenCLSDK/opencl-sdk-1.2.2)' >> build_android/config.cmake

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/home/avoronov/Android/Sdk/ndk/21.4.7075529/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=android-28 -DANDROID_STL=c++_static ../tvm
make -j16
```

# Pre-launch preparation 
#### Android device preparation 
- Turn on developer mode
- Enable usb debugging
- Connect with your desktop
- Select USB configuration(for example: File transfer / Android Auto)

Helper: https://developer.android.com/studio/debug/dev-options#enable

#### Connection preparation
- Get android device id using `adb`
```
adb devices
# List of devices attached
# 10e695e6
```
`10e695e6` for example
- Make sure /data/local/tmp is clear on android device using `adb`
```
adb shell
rm -rf /data/local/tmp/*
```
- Copy required binaries from build_android directory using `adb`
```
adb -s 10e695e6 push {libtvm_runtime.so,tvm_rpc} /data/local/tmp
```
- Connect desktop to wifi
- Connect android device to the same wifi
- Find wifi using `iwconfig`:
```
iwconfig
# lo        no wireless extensions.
# 
# eno1      no wireless extensions.
# 
# wlo1      IEEE 802.11  ESSID:"IceNetAp5"  
#           Mode:Managed  Frequency:5.18 GHz  Access Point: C4:AD:34:FB:24:B5   
#           Bit Rate=173.3 Mb/s   Tx-Power=22 dBm   
#           Retry short limit:7   RTS thr:off   Fragment thr:off
#           Power Management:on
#           Link Quality=57/70  Signal level=-53 dBm  
#           Rx invalid nwid:0  Rx invalid crypt:0  Rx invalid frag:0
#           Tx excessive retries:0  Invalid misc:31   Missed beacon:0
```
`wlo1` for example
- Find wifi ip address using `ifconfig`
```
ifconfig
# eno1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
#         ether 36:2b:ba:21:bb:39  txqueuelen 1000  (Ethernet)
#         RX packets 0  bytes 0 (0.0 B)
#         RX errors 0  dropped 0  overruns 0  frame 0
#         TX packets 0  bytes 0 (0.0 B)
#         TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
# 
# lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
#         inet 127.0.0.1  netmask 255.0.0.0
#         inet6 ::1  prefixlen 128  scopeid 0x10<host>
#         loop  txqueuelen 1000  (Local Loopback)
#         RX packets 3205  bytes 42031310 (42.0 MB)
#         RX errors 0  dropped 0  overruns 0  frame 0
#         TX packets 3205  bytes 42031310 (42.0 MB)
#         TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
# 
# wlo1: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
#         inet 192.168.1.5  netmask 255.255.255.0  broadcast 192.168.88.255
#         inet6 fad3::d9ae:327bc:3b3f:5bce  prefixlen 64  scopeid 0x20<link>
#         ether 2a:48:6e:4c:4d:33  txqueuelen 1000  (Ethernet)
#         RX packets 804090  bytes 1188439538 (1.1 GB)
#         RX errors 0  dropped 0  overruns 0  frame 0
#         TX packets 210296  bytes 55854116 (55.8 MB)
#         TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

```
`192.168.1.5` for example


# Run console 1
Example:
```
export PATH=$PATH:/home/avoronov/repo/deelvin_qualcomm/build_host
export PYTHONPATH=$PYTHONPATH:/home/avoronov/repo/deelvin_qualcomm/tvm/python
python3 -m tvm.exec.rpc_tracker --host 192.168.1.5 --port 9190 --port-end 9192
```

# Run console 2
Example:
```
/home/avoronov/repo/deelvin_qualcomm/scripts/launch_rpc.sh -d 10e695e6 -t 192.168.1.5:9190 -k android -p /data/local/tmp
```

# Run console 3
Example:
```
export TVM_TRACKER_HOST=192.168.1.5
export TVM_TRACKER_PORT=9190
export TVM_NDK_CC=/home/avoronov/Android/Sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++
export PATH=$PATH:/home/avoronov/repo/deelvin_qualcomm/build_host
export PYTHONPATH=$PYTHONPATH:/home/avoronov/repo/deelvin_qualcomm/tvm/python

# check that the device is available
python3 -m tvm.exec.query_rpc_tracker --port 9190 --host 192.168.1.5
```

#### To run fp16acc16
Comment strategy.add_implementation on these lines:
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L47
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L62
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L105
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L121
```
python /home/avoronov/repo/deelvin_qualcomm/scripts/evaluate.py -m mace_deeplabv3 -t float16 -k android --target="opencl --device=adreno"  -l ./tune/mace_deeplabv3.texture.float16acc16.autotvm_11.log > log.txt 2>&1
```
#### To run fp16acc32
Comment strategy.add_implementation on these lines:
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L40
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L55
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L98
https://github.com/Deelvin/qualcomm/blob/master/tvm/python/tvm/relay/op/strategy/adreno.py#L114

```
python3 /home/avoronov/repo/deelvin_qualcomm/scripts/evaluate.py -m mace_deeplabv3 -t float16 -k android --target="opencl --device=adreno" -l ./tune/mace_deeplabv3.texture.float16acc32.autotvm_11.log > log.txt 2>&1
```
#### To run fp32acc32
Nothing to comment
```
python3 /home/avoronov/repo/deelvin_qualcomm/scripts/evaluate.py -m mace_deeplabv3 -t float32 -k android --target="opencl --device=adreno" -l ./tune/mace_deeplabv3.texture.float32.autotvm_11.log > log.txt 2>&1
```

# Notes
Turn on tuning
```
--tune
```
Turn on debugging
```
--debug
```
Modify the number of tuning attempts
```
n_trial=333
```
Modify the number of model runs in the benchmark measurement
```
num_iter=200
```
Stabilize and save the output 
```
def _benchmark(
+    np.random.seed(1)
...
    print("%g secs/iteration\n" % cost)
+    np.set_printoptions(threshold=sys.maxsize)
+    num_outputs = m.get_num_outputs()
+    outputs = []
+    for i in range(num_outputs):
+        tvm_output = m.get_output(i)
+        outputs.append(tvm_output.asnumpy())
+    with open("output.txt", "w" ) as f:
+        f.write(str(outputs))
    if validator:
```

