# Face mask detector : COVID-19 prevention

This github inspired by the work of Adrian Rosebrock of [Pyimagesearch](https://www.pyimagesearch.com/) for the detection of faces and identification of the wear of face masks.

The current project offers a multi camera proposition to have a detection over several point of view.

# Install

Python 3 is required. Python 2 is not supported.
Install the required packages with the command :


```sh
pip install -r requirements.txt
```

# Organisation

- **IP_cameras_client_side.py**

The particularity of this code compared to the one available at PyimageSearch is a slightly different neural network (which will be discussed in the training part of the algorithm) and the possibility of connecteing several cameras. In this case, we use IP cameras or cameras connected to a Raspberry pi. Those cameras will send their images to our main computer that will process them.

- **detect_mask_video_IP.py**

This function runs the neural network on your cameras coming from the computer, the IP cameras and the raspberry pi.

Disclaimer: In case you are using the Rapsberry Pi and IP cameras, the previous script should be running on each of the devices FIRST.


# Camera IP, Raspberry Pi Camera

1 - To use IP cameras, you need to run the following script for each camera on your computer while replacing the IP address (They should be on the same network as your computer).
In order to test the IP camera implementation without owning any IP camera, you can install an application such as [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en) on your phone. 

The command should will be : 

´´´sh
python IP_cameras_client_side.py -t IP -p rtsp://10.153.3.159:8080///h264_ulaw.sd
´´´

were the -t indicate the type of camera (in this case IP) and the -p indicates the port of the camera.

2 - To use Raspberry Pi camera modules, you need to run the following code on each of them (while they are connected to the same network as your computer).

The command is :

```sh
python IP_cameras_client_side.py -t PI
```

where -t indicates the type of camera (in this case PI)



# Detection part

The command should be : 

```sh
python detect_mask_video_IP.py -c 0.5  -s 500 -d 0 1 -a -nc 1
```

Where :

-c is the minimum probability to filter weak detections

-s is the width of the windows 

-d is the cameras connected physically to your device (in this case, the camera 0 and 1 were used)

-nc is the number of both IP and raspberry pie camera that you have running on the same network

-a is an option to hide the face of the individuals (let's protect our privacy, at least a bit).

For those who have been reading until the end, if you do not remember what each parameter do, just add --help at the end of your commands.


# Training

This implementation does not come yet with a training routine. It is a work in progress and this part will be updated in the following commits.