# Generating TensorRT engine files for TAO Pretrained Models using the TAO converter

## Installation
Follow instructions for [installing the TAO Converter](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter).

## Engine file generation
Use the given commands to generate engine files for each model. You may change/specify the Batch Size with the **-b** argument. Information on other required and optional arguments can be found under **Running the TAO Converter** [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter).

## [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)
1. Download model and calibration files from Nvidia NGC:
- **resnet34_peoplenet_int8.etlt:**
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.5/files/resnet34_peoplenet_int8.etlt'
```
- **resnet34_peoplenet_int8.txt:**
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/deployable_quantized_v2.5/files/resnet34_peoplenet_int8.txt'
```

2. Generate the GPU engine file:
```
tao-converter -k tlt_encode \
                   -d 3,960,544 \
                   -o output_cov/Sigmoid,output_bbox/BiasAdd \
                   -c <path-to>/resnet34_peoplenet_int8.txt \
                   -e <path-to-store-generated-engine>/peoplenet_gpu.engine \
                   -b 32 \
                   -m 32 \
                   -t int8 <path-to>/resnet34_peoplenet_int8.etlt

```  

3. Generate the DLA engine file:
```
tao-converter -k tlt_encode \
                   -d 3,960,544 \
                   -o output_cov/Sigmoid,output_bbox/BiasAdd \
                   -c <path-to>/resnet34_peoplenet_int8.txt \
                   -e <path-to-store-generated-engine>/peoplenet_dla_0.engine \
                   -b 4 \
                   -m 4 \
                   -u 0 \
                   -t int8 <path-to>/resnet34_peoplenet_int8.etlt

```

## [ActionRecognitionNet 2D](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet)
1. Download model from Nvidia NGC:
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/actionrecognitionnet/versions/deployable_v1.0/files/resnet18_2d_rgb_hmdb5_32.etlt'
```

2. Generate the GPU engine file:
```
tao-converter <path-to>/resnet18_2d_rgb_hmdb5_32.etlt -d 96x224x224 -k nvidia_tao -p input_rgb,64x96x224x224,64x96x224x224,64x96x224x224 -t fp16  -e <path-to-store-generated-engine>/action_recog_2d_gpu.engine
```

## [ActionRecognitionNet 3D](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/actionrecognitionnet)
1. Download model from Nvidia NGC:
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/actionrecognitionnet/versions/deployable_v1.0/files/resnet18_3d_rgb_hmdb5_32.etlt'
```

2. Generate the GPU engine file:
```
tao-converter -e <path-to-store-generated-engine>/action_recog_3d_gpu.engine -k nvidia_tao -p input_rgb,32x3x32x224x224,32x3x32x224x224,32x3x32x224x224 -t fp16 <path-to>/resnet18_3d_rgb_hmdb5_32.etlt
```

## [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet)
1. Download model and calibration files from Nvidia NGC:
- **resnet18_dashcamnet_pruned.etlt:**
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.1/files/resnet18_dashcamnet_pruned.etlt'
```
- **dashcamnet_int8.txt:**
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/dashcamnet/versions/pruned_v1.0.1/files/dashcamnet_int8.txt'
```

2. Generate the GPU engine file:
```
tao-converter -k tlt_encode \
                   -d 3,960,544 \
                   -o output_cov/Sigmoid,output_bbox/BiasAdd \
                   -c <path-to>/dashcamnet_int8.txt \
                   -e <path-to-store-generated-engine>/dashcamnet_gpu.engine \
                   -b 32 \
                   -m 32 \
                   -t int8  \  
                   <path-to>/resnet18_dashcamnet_pruned.etlt
```

3. Generate the DLA engine file:
```
tao-converter -k tlt_encode \
                   -d 3,960,544 \
                   -o output_cov/Sigmoid,output_bbox/BiasAdd \
                   -c <path-to>/dashcamnet_int8.txt \
                   -e <path-to-store-generated-engine>/dashcamnet_dla_0.engine \
                   -b 8 \
                   -m 8 \
                   -u 0 \
                   -t int8 \
                   <path-to>/resnet18_dashcamnet_pruned.etlt
```

## [BodyPoseNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/bodyposenet)
1. Download model and calibration files from Nvidia NGC:
- **model.etlt:**
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/bodyposenet/versions/deployable_v1.0.1/files/model.etlt'
```
- **int8_calibration_288_384.txt:**
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/bodyposenet/versions/deployable_v1.0.1/files/int8_calibration_288_384.txt'
```

2. Generate the GPU engine file:
```
tao-converter -k nvidia_tlt \
                    -p input_1:0,1x288x384x3,64x288x384x3,64x288x384x3 \
                    -c <path-to>/int8_calibration_288_384.txt \
                    <path-to>/model.etlt \
                    -t int8 \
                    -e <path-to-store-generated-engine>/bodyposenet_gpu.engine
```

## [LPRNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/lprnet)
1. Download model from Nvidia NGC:
```
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/lprnet/versions/deployable_v1.0/files/us_lprnet_baseline18_deployable.etlt'
```

2. Generate the GPU engine file:
```
tao-converter -k nvidia_tlt -p image_input,128x3x48x96,128x3x48x96,128x3x48x96 <path-to>/us_lprnet_baseline18_deployable.etlt -t fp16 -e <path-to-store-generated-engine>/lpr_us_gpu.engine
```
