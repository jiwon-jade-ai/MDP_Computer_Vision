#! /usr/bin/python

import cv2
import torch
import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from torchvision import transforms as T
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


percentage = 0
MODEL_PATH = "Unet-diagdataset.pt"
model = torch.load(MODEL_PATH)
image_topic =  "/zed2/zed_node/rgb/image_rect_color"
road_percentage_pub = rospy.Publisher('cv_nav/road_percentage', Float64, queue_size=1)
road_percentage_msg = Float64()

def callback_image(msg):
    global percentage
    global road_percentage_pub
    global road_percentage_msg
    global model
    print("Received an image!")
    cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    image_resize = cv2.resize(cv2_img, (640, 384))
    pred_mask = predict_image_mask_miou(model, image_resize)
    cimage = pred_mask[248:328, 35:385] / 85 # sidewalk is 85
    count = (cimage == 1.0).sum()
    percentage = float(count / np.array(cimage).size * 100)
    road_percentage_msg.data = percentage
    road_percentage_pub.publish(road_percentage_msg)


def predict_image_mask_miou(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)

    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked


if __name__ == '__main__':
    rospy.init_node('segmenation_node')
    #global image_topic
    image_subscriber = rospy.Subscriber(image_topic, Image, callback_image)
    rospy.spin()


# load model
# model.eval
