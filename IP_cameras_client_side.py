from imutils.video import VideoStream
import imagezmq
import socket
# change this to your stream address
path = "rtsp://10.153.3.159:8080///h264_ulaw.sdp"
cap = VideoStream(path)

# change this to your server address
sender = imagezmq.ImageSender(connect_to='tcp://localhost:5555') 

cam_id = socket.gethostname()
stream = cap.start()


while True:
    frame = stream.read()
    sender.send_image(cam_id, frame)