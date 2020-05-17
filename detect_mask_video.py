import os 
import argparse
from IP_cameras_client_side import client_cam
from detect_mask_video_IP import face_detector
from imutils.video import VideoStream

import concurrent.futures
import time


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,default="face_detector",
	            help="path to face detector model directory"),
ap.add_argument("-s", "--size", type=int,default=400,
	            help="Size of the windows to process"),
ap.add_argument("-m", "--model", type=str,
	            default="mask_detector.model",help="path to trained face mask detector model"),
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	            help="minimum probability to filter weak detections"),
ap.add_argument("-d", "--devices",  nargs='+', help = "Cameras connected to the computer",
                default=[], type=int),
ap.add_argument("-se", "--sender", type=str,default="tcp://localhost:5555",
                help="Send the images to (default tcp://localhost:5555)"),
ap.add_argument("-ts", "--types", help="types of the camera [IP or PI cam]", 
                type =str,nargs="+" ,default = []),
ap.add_argument("-ps", "--paths", help = "paths to the IP cam (for the PI cam, put whatever value, really)",
                type= str, default=[]),
ap.add_argument("-a", "--anonymity",
		        help="anonymity function for the user's victim", action='store_true')

args = vars(ap.parse_args())

numberCam = len(args["types"])
args["number_cam"]=numberCam

print(numberCam)

arguments = []

for camType, camPath  in zip(args["types"], args["paths"]):
    args["type"] = camType
    args["path"] = camPath
    arguments.append(args)

 
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = []
    
    for argument in arguments:
        results.append(executor.submit(client_cam, argument))    
    print(results)
    face_detector(args)

