# ADI Neural Network Library

This is the repository for the Neural Network primitives optimized for ADI Devices. 

### Headless build and flash
This project allows headless building.
The commands are bash commands and need to be run in such commandline tools. Tested the commands using git bash.
To build, run the following from `adi_sharcfx_nn\Project` folder:
```
make
```

Note that the `libadi_sharcfx_nn.a` static library to be built first before building TFLM library or the application.

To clean the built objects:
```
make clean
```

#### Building on CCES 3.0.2 and beyond
To build with CCES > 3.0.2,

```
cd adi_sharcfx_nn\Project
make SHARCFX_ROOT=/c/analog/cces/3.0.2
cd kws_realtime
make flash SHARCFX_ROOT=/c/analog/cces/3.0.2
```
Build all the library and application code with the same toolchain.

#### Troubleshooting

Incase of the following error:
1. 
```
bash: make: command not found
```
Use:
```
/c/analog/cces/<cces_version>/make.exe <command>
```

2. To choose between Release build or Debug build, use

```
make CONFIG=<Release/Debug>
```
