#! /usr/bin/python

import cv2
import torch
import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from torchvision import transforms as T
from cv_bridge import CvBridge, CvBridgeError

# Model
bridge = CvBridge()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "Unet-diagdataset.pt"
model = torch.load(MODEL_PATH)

# Topic publishers
forward_box_pub = rospy.Publisher('cv_nav/road_percentage/forward', Float64, queue_size=1)
rotate_box_pub = rospy.Publisher('cv_nav/road_percentage/rotate', Float64, queue_size=1)
right_box_pub = rospy.Publisher('cv_nav/road_percentage/right', Float64, queue_size=1)
left_box_pub = rospy.Publisher('cv_nav/road_percentage/left', Float64, queue_size=1)
image_topic =  "/zed/zed_node/rgb/image_rect_color"
road_percentage_msg = Float64()


# Bounding box metadata. Each bounding box is described by [publisher, x_min, x_max, y_min, y_max]
forward_box_info = [forward_box_pub, 248, 328, 35, 385] # 250 391 250 320
rotate_box_info = [rotate_box_pub, 160, 481, 270, 311]
right_box_info = [right_box_pub, 440, 641, 100, 281]
left_box_info = [left_box_pub, 0, 201, 100, 281]
bounding_boxes = [forward_box_info, rotate_box_info, right_box_info, left_box_info]


def callback_image(msg):
    global model
    global road_percentage_msg
    print("Received an image!")
    cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    image_resize = cv2.resize(cv2_img, (640, 384))
    pred_mask = predict_image_mask_miou(model, image_resize)
    for bounding_box in bounding_boxes:
        publisher, x_min, x_max, y_min, y_max = bounding_box
        cimage = pred_mask[x_min:x_max, y_min:y_max] / 85 # sidewalk is 85
        count = (cimage == 1.0).sum()
        road_percentage_msg.data = float(count / np.array(cimage).size * 100)
        publisher.publish(road_percentage_msg)

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
    sub = rospy.Subscriber(image_topic, Image, callback_image)
    rospy.spin()


# load model
# model.eval
