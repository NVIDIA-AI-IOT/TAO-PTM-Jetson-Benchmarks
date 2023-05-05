# TAO-PTM-Jetson-Benchmarks
[TAO](https://developer.nvidia.com/tao-toolkit) Pretrained Models Benchmarking on Jetson

This project provides steps for benchmarking on the following models on Jetson devices:
- [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)
- [Action Recognition 2D](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet)
- [Action Recognition 3D](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet)
- [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet)
- [BodyPoseNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodyposenet)
- [LPRNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lprnet)

### Install Requirements
```
git clone https://github.com/NVIDIA-AI-IOT/TAO-PTM-Jetson-Benchmarks.git
cd TAO-PTM-Jetson-Benchmarks 
sudo sh install_requirements.sh
```
Note: All libraries will be installed for ```python3```

### Follow instructions in [engine-generation](https://github.com/NVIDIA-AI-IOT/TAO-PTM-Jetson-Benchmarks/blob/main/docs/engine-generation.md) to install the TAO Converter, download Pretrained models and generate TensorRT engine files. 

Modify the appropriate benchmark csv file to use the same batch sizes used during engine generation. For example, let's say you are benchmarking on a Jetson Orin and specified **64** as the GPU batch size while generating the engine file for Action Recognition 2D in the step above. In this case, [orin_ptm.csv](https://github.com/NVIDIA-AI-IOT/TAO-PTM-Jetson-Benchmarks/blob/main/benchmark_csv/orin_ptm.csv) should have the number **64** under **BatchSizeGPU** for action_recog_2d. The **input** column should also incorporate the batch size accordingly, for example, `input_rgb:BatchSizex96x224x224`. 

## Running Benchmarks for Jetson Orin

``` sudo python3 benchmark.py --all --csv_file_path <path-to>/benchmark_csv/orin_ptm.csv --model_dir <absolute-path-to-engine-files>```  <br /> 

#### Running Individual Benchmark Models

Use the `--model_name` argument to specify an individual model for benchmarking:
| Model | --model_name argument|
| ------ | ------ |
|  PeopleNet      |    peoplenet    |
|  Action Recognition 2D      |   action_recog_2d     |
|  Action Recognition 3D      |   action_recog_3d     |
|  DashCamNet      |   dashcamnet     |
|  BodyPoseNet      |    bodyposenet    |
|  LPRNet      |    lpr_us    |

For example, for running only PeopleNet on Jetson Orin:
``` sudo python3 benchmark.py --model_name peoplenet --csv_file_path <path-to>/benchmark_csv/orin_ptm.csv --model_dir <absolute-path-to-engine-files>```  <br />

## Running Benchmarks for Jetson AGX Xavier

``` sudo python3 benchmark.py --all --csv_file_path <path-to>/benchmark_csv/agx_xavier_ptm.csv --model_dir <absolute-path-to-engine-files>```

Follow [Running Individual Benchmark Models](https://github.com/NVIDIA-AI-IOT/TAO-PTM-Jetson-Benchmarks#running-individual-benchmark-models) to specify a single model for benchmarking.

## Running Benchmarks for Jetson Orin Nano

``` sudo python3 benchmark.py --all --csv_file_path <path-to>/benchmark_csv/orin_nano_ptm.csv --model_dir <absolute-path-to-engine-files>```

Follow [Running Individual Benchmark Models](https://github.com/NVIDIA-AI-IOT/TAO-PTM-Jetson-Benchmarks#running-individual-benchmark-models) to specify a single model for benchmarking.

## Running Benchmarks for Jetson Nano

``` sudo python3 benchmark.py --all --csv_file_path <path-to>/benchmark_csv/nano_ptm.csv --model_dir <absolute-path-to-engine-files>```

Follow [Running Individual Benchmark Models](https://github.com/NVIDIA-AI-IOT/TAO-PTM-Jetson-Benchmarks#running-individual-benchmark-models) to specify a single model for benchmarking.

