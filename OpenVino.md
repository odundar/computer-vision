## Openvino - Open Visual Inference & Neural Network Optimization (OpenVINO™)

A step-by-step, comprehensive Intel OpenVINO tutorial.

I would like to go with a step by step approach to cover most popular features a developer can benefit from OpenVINO therefore I should first address:

- What is Intel OpenVINO?
- What kind tools OpenVINO provides for developers?
- What processes are being improved while developing?
- How can I go to production level code with Intel OpenVINO.

## Introduction

Intel(R) OpenVINO(TM) (Open Visual Inference Neural Optimisations)

OpenVINO is a set of software tools to enhance computer vision application development with following components:

- Deep Learning Model Optimizer
- Deep Learning Inference Engine
- Drivers and runtimes for OpenCL version 2.1
- Intel Media SDK
- OpenCV
- OpenVX

All the components provided are optimized for Intel CPU, GPU, FPGA's and Movidius Neural Compute Stick

## Installation

It is suggested to install OpenVino to a separate development environment to make things easier for your side, otherwise version mismatches and PATH variables can override themselves.

You can follow installation instructions from following URL: 

#####Linux

https://software.intel.com/en-us/articles/OpenVINO-Install-Linux

#####Windows

https://software.intel.com/en-us/articles/OpenVINO-Install-Windows

```text
After installation one important part to remember is to set environment variables.

For Linux:

setupvars.sh file located under /opt/intel/computer_vision/bin folder (/opt/intel installation directory) .


$ source setupvars.sh


You need to repeat these at each time you load a new terminal to work on demo, samples etc. We will see how we include libraries etc in next sections.

For Windows:



TODO:

```

After the installation, you should verify your setup was complete and functioning.

```commandline

source /opt/intel/computer_vision/bin/setupvars.sh

cd /opt/intel/computer_vision/deployment_tools/demo

sudo ./demo_squeezenet_download_convert_run.sh

```

You should see that, script downloading models accordingly and running classification sample for the image inside the folder. End of the execution output should like below:

```commandline
Image /opt/intel/computer_vision_sdk_2018.2.300/deployment_tools/demo/../demo/car.png

817 0.8363345 label sports car, sport car
511 0.0946488 label convertible
479 0.0419131 label car wheel
751 0.0091071 label racer, race car, racing car
436 0.0068161 label beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
656 0.0037564 label minivan
586 0.0025741 label half track
717 0.0016069 label pickup, pickup truck
864 0.0012027 label tow truck, tow car, wrecker
581 0.0005882 label grille, radiator grille

[ INFO ] Execution successful

```

## Model Downloader

Model Downloader is a Python script located under ```/opt/intel/computer_vision_sdk/deployment_tools/model_downloader```
 to help fetching popular public deep learning topologies and prepare models.

Their license information for models can be seen under same directory.

Aim of this tool is to make a quick start for engineers who is starting to work on computer vision with OpenVINO and let them practive model optimisations.

Below is the list of models can be downloaded. These are either ```Caffe, Tensorflow or MxNet``` models which are ready to be optimized.

```commandline
cd /opt/intel/computer_vision_sdk_2018.2.300/deployment_tools/model_downloader 
python3 downloader.py --print_all
densenet-121
densenet-161
densenet-169
densenet-201
squeezenet1.0
squeezenet1.1
mtcnn-p
mtcnn-r
mtcnn-o
mobilenet-ssd
vgg19
vgg16
ssd512
ssd300
inception-resnet-v2
dilation
```

When you download models, they are being placed under same folder according to their use ```classification, object_detection, semantic_segmentation```

Note: using sudo since files are downloaded under `opt` folder.

```commandline

$ sudo python3 downloader.py --name ssd512

###############|| Start downloading models ||###############

###############|| Start downloading weights ||###############

###############|| Start downloading topologies in tarballs ||###############

...100%, 98624 KB, 1967 KB/s, 50 seconds passed ========= ssd512.tar.gz ====> /opt/intel/computer_vision_sdk_2018.2.300/deployment_tools/model_downloader/object_detection/common/ssd/512/caffe/ssd512.tar.gz


###############||  Post processing ||###############

========= Extract files from ssd512.tar.gz
========= Move ssd512.prototxt and ssd512.caffemodel to /opt/intel/computer_vision_sdk_2018.2.300/deployment_tools/model_downloader/object_detection/common/ssd/512/caffe after untarring archive =========
========= Deleting "save_output_param" from ssd512.prototxt  =========

```

### Pretrained Free Models

Together with public models which can be downloaded with ```model_downloader```, Intel OpenVINO has a set of pretrained Deep Learning models, which are ready to be used.

See list of models from the following URL:
 
https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models

Models are placed under folder, intel_models. Each model have .bin and .xml (Intermediate Representation file)files which are ready to be used for OpenVINO applications.

Public Caffe, Tensorflow and MxNet models should be converted to .bin and .xml files before using them with OpenVINO. 

```commandline
$ cd /opt/intel/computer_vision_sdk_2018.2.300/deployment_tools/intel_models
ls
age-gender-recognition-retail-0013  head-pose-estimation-adas-0001  license-plate-recognition-barrier-0001        person-detection-retail-0013                  semantic-segmentation-adas-0001
emotions-recognition-retail-0003    index.html                      pedestrian-and-vehicle-detector-adas-0001     person-reidentification-retail-0076           vehicle-attributes-recognition-barrier-0039
face-detection-adas-0001            intel_logo.png                  pedestrian-detection-adas-0002                person-reidentification-retail-0079           vehicle-detection-adas-0002
face-detection-retail-0004          intel_styles.css                person-attributes-recognition-crossroad-0031  person-vehicle-bike-detection-crossroad-0078  vehicle-license-plate-detection-barrier-0007
face-person-detection-retail-0002   License.pdf                     person-detection-retail-0001                  road-segmentation-adas-0001

$ cd person-detection-retail-0001/
$ ls
cnn.prototxt  description  FP16  FP32
$ ls FP16
person-detection-retail-0001.bin  person-detection-retail-0001.xml

```

We will try to use a public model and a free model to complete our application.

List of models:

Face Detection Models

- Standard Model 
- Enhanced Model
- Retail Environment Model
- Head Position
- Emotion Recognition

Human Detection Models

- Eye-level Detection
- High-Angle Detection
- Detect People, Vehicles & Bikes
- Pedestrian Detection
- Pedestrian & Vehicle Detection
- Identify Someone in Different Videos
- Pedestrian Attributes

Vehicle Feature Detection

- Vehicle Detection
- License Plate Detection: Small-Footprint Network
- License Plate Detection: Front-Facing Camera
- Vehicle Metadata
- Identify RoadSide Objects
- Advanced Roadside Identification
 

## Model Optimizer

OpenVINO Model Optimizer is a set of tools which are able to optimize existing Caffe, Tensorflow, MxNet DL models to Intermediate Representative (IR) files.

Intermediate Representative files are .xml and .bin files respectively:

- .xml file: Describes the network topology
- .bin file: Contains the weights and biases binary data

Model Optimizer binary and script placed under ```/opt/intel/computer_vision_sdk_2018.2.300/deployment_tools/model_optimizer``` folder.

Before running model optimizer make sure all prerequisets have been installed.

```commandline
cd install_prerequisites

sudo ./install_prerequisites.sh

```

Now, model optimizer is ready to be used.

### How Model Optimizer Work

Model optimiser uses certain optimisation techniques which are scientifically proven that, optimizes, accelerates neural network in many ways memory usage, less operations, performance etc. without any lose of functionality and accuracy.

Techniques are listed in documentation `deployment_tools/documentation/MOTechniques.html`.

 

### Optimisation Techniques Used in Model Optimiser


### Custom Layer to DL Model


## Inference Engine


## Application Development

Use case:

We want to check identities of people and vehicles from a live stream working on the edge and send over meta data; face data and vehicle plate data.

We will use:

- OpenVino Inference Engine for inference 

- Media SDK to decode incomings stream

- Transport meta data over network.

- Determine performance and log collection properties.

- Determine configuration parameters for the application.

- OpenCV to show post processed images

We don't need encoding since we will not stream the live video to any other source, and keep things little less complicated.

Note that, at this level, this is just a demonstration application without any optimization considerations.

### System Design

People and Vehicle Identifier Application on the Edge

![Edge CV Application](resources/edge_cv_application.png)

As seen from the above design and flow diagram:

1. Application will read video frames from source, parameters will be loaded.

2. Media SDK will be used for decode and pre-processes to make input ready for inference.

3. OpenVINO IE will load models and use the frames to infer required data.

4. Finally, OpenCV will be used to display post processed output for demo purposes.

5. Log and performance analyzer will be embedded into system for detailed analysis.







 