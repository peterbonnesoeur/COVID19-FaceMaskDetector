import cv2 
import imagezmq

image_hub = imagezmq.ImageHub()

while True:  
    cam_id, frame = image_hub.recv_image()
    
    print(cam_id)

    cv2.imshow(cam_id, frame)  

    cv2.waitKey(1)

    image_hub.send_reply(b'OK')