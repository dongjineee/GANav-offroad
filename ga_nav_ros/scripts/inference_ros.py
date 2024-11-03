#!/usr/bin/env python3
import os
import cv2
import torch
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageSegmentationNode:
    def __init__(self):
        config_path = rospy.get_param('~config_path', '../configs/ours/ganav_group6_lake.py')
        checkpoint_path = rospy.get_param('~checkpoint', '../work_dirs/ganav_group6_lake/latest.pth')
        palette = rospy.get_param('~palette', 'lake_group')
        device = rospy.get_param('~device', 'cuda:0')
        resize_width = rospy.get_param('~resize_width', 680)
        resize_height = rospy.get_param('~resize_height', 480)
        opacity = rospy.get_param('~opacity', 0.5)

        self.palette = get_palette(palette)
        self.model = init_segmentor(config_path, checkpoint_path, device=device)
        self.resize = (resize_width, resize_height)
        self.opacity = opacity
        self.bridge = CvBridge()
        
        rospy.Subscriber("/realsense/color/image_raw", Image, self.image_callback)
        self.result_pub = rospy.Publisher("/segmentation_result", Image, queue_size=1)

    def load_and_resize_image(self, img_msg):
        """Convert ROS image message to OpenCV format and resize."""
        img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        height, width = img.shape[:2]
        ##If resizing the image is necessary
        # resized_img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        # return resized_img
        return img

    def publish_segmentation_result(self, img):
        """Perform segmentation on the image and publish the result."""
        result = inference_segmentor(self.model, img)
        img_result = self.model.show_result(img, result, palette=self.palette, show=False, opacity=self.opacity)
        ##If saving the image necessary
        # save_path = os.path.join('/root/result', 'result.png')
        # cv2.imwrite(save_path, img_result)
        # rospy.loginfo(f"Image saved to {save_path}")
        # Convert OpenCV image to ROS Image message and publish
        result_msg = self.bridge.cv2_to_imgmsg(img_result, encoding="bgr8")
        self.result_pub.publish(result_msg)
        rospy.loginfo("Published segmentation result")

    def image_callback(self, img_msg):
        img = self.load_and_resize_image(img_msg)
        self.publish_segmentation_result(img)

def main():
    rospy.init_node('inference_ros')
    node = ImageSegmentationNode()
    rospy.spin()

if __name__ == '__main__':
    main()
