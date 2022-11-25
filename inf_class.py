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

MODEL_PATH = "Unet-diagdataset.pt"
model = torch.load(MODEL_PATH)
image_topic =  "/zed/zed_node/rgb/image_rect_color"

def create_callback(x_min, x_max, y_min, y_max, topic_name):
    road_percentage_pub = rospy.Publisher(topic_name, Float64, queue_size=1)
    road_percentage_msg = Float64()
    def callback_image(msg):
        global model
        nonlocal road_percentage_pub
        nonlocal road_percentage_msg
        print("Received an image!")
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_resize = cv2.resize(cv2_img, (640, 384))
        pred_mask = predict_image_mask_miou(model, image_resize)
        cimage = pred_mask[x_min:x_max, y_min:y_max] / 85 # sidewalk is 85
        count = (cimage == 1.0).sum()
        road_percentage_msg.data = float(count / np.array(cimage).size * 100)
        road_percentage_pub.publish(road_percentage_msg)
    return callback_image

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
    box_callback_main = create_callback(248, 328, 35, 385, 'cv_nav/road_percentage/main')
    box_callback_rotate = create_callback(200, 368, 300, 385, 'cv_nav/road_percentage/rotate')
    box_callback_left = create_callback(0, 320, 100, 300, 'cv_nav/road_percentage/left')
    box_callback_right = create_callback(320, 640, 100, 300, 'cv_nav/road_percentage/right')

    main_subscriber = rospy.Subscriber(image_topic, Image, box_callback_main)
    rotate_subscriber = rospy.Subscriber(image_topic, Image, box_callback_rotate)
    left_subscriber = rospy.Subscriber(image_topic, Image, box_callback_left)
    right_subscriber = rospy.Subscriber(image_topic, Image, box_callback_right)
    rospy.spin()


# load model
# model.eval
