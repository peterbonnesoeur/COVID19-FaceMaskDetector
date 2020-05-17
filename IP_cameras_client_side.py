# run this program on each RPi or on your computer for the Ip cams to send a labelled image stream
import socket
import time
from imutils.video import VideoStream
import imagezmq
import argparse


def client_cam(args):

    sender = imagezmq.ImageSender(connect_to=args["sender"])
    rpi_name = socket.gethostname() # send unique RPi hostname with each image

    if args["type"] == "PI": #If you are using a picam, use the following:
        picam = VideoStream(usePiCamera=True).start()
    elif args["type"] == "IP": #Else, do this :
        path = args["paths"]#"rtsp://10.153.3.159:8080///h264_ulaw.sdp"
        picam = VideoStream(path).start()
    else: 
        print("Not defined camera type")

    time.sleep(2.0)  # allow camera sensor to warm up
    while True:  # send images as stream until Ctrl-C
        image = picam.read()
        sender.send_image(rpi_name, image)
    
if __name__ == "__main__":
    
    print("here2")
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-se", "--sender", type=str,
        default="tcp://localhost:5555",
        help="Send the images to (default tcp://localhost:5555)"),
    ap.add_argument("-t", "--type", 
                    help="type of the camera [IP or PI cam]", 
                    type =str, default = "None"),
    ap.add_argument("-p", "--path",
                    help = "path to the IP cam",
                    type= str, default="rtsp://10.153.3.159:8080///h264_ulaw.sdp"),
    args = vars(ap.parse_args())
    
    client_cam(args)