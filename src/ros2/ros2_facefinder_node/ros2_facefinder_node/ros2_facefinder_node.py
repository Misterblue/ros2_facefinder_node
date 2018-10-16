# Copyright 2018 Robert Adams
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import queue
import threading
import time
import sys

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray

import dlib
import imageio

class ROS2_facefinder_node(Node):

    def __init__(self):
        super().__init__('ros2_facefinder_node', namespace='raspicam')

        self.param_image_input_topic = self.get_parameter_or('image_input_topic', 'raspicam_compressed')
        self.param_image_compressed = self.get_parameter_or('image_compressed', True)
        self.param_face_output_topic = self.get_parameter_or('face_output_topic', 'found_faces')

        self.initialize_face_recognizer()
        self.initialize_image_subscriber()
        self.initialize_bounding_box_publisher()
        self.initialize_processing_queue()

    def destroy_node(self):
        # overlay Node function called when class is being stopped and camera needs closing
        super().destroy_node()

    def initialize_face_recognizer(self):
        # Get  the detector that will be used herein
        # Will eventually add face recognizer but this is a start
        self.detector = dlib.get_frontal_face_detector()

    def initialize_image_subscriber(self):
        # Setup subscription for incoming images
        if self.param_image_compressed:
            self.receiver = self.create_subscription(
                        CompressedImage, self.param_image_input_topic, self.receive_image)
        else:
            self.receiver = self.create_subscription(
                        Image, self.param_image_input_topic, self.receive_image)
        self.frame_num = 0

    def initialize_bounding_box_publisher(self):
        self.bounding_box_publisher = self.create_publisher(Int32MultiArray, self.param_face_output_topic)

    def initialize_processing_queue(self):
        # Create a queue and a thread that processes messages in the queue
        self.queue_lock = threading.Lock()

        self.image_queue = queue.Queue()
        # self.image_queue = queue.SimpleQueue()  # introduced in Python 3.7

        # thread to read images placed in the queue and process them
        self.processor_event = threading.Event()
        self.processor = threading.Thread(target=self.process_images, name='facefinder')

        self.processor.start()

    def stop_workers(self):
        # if workers are initialized and running, tell them to stop and wait until stopped
        if hasattr(self, 'processor_event') and self.processor_event != None:
            self.processor_event.set()
        if hasattr(self, 'processor') and self.processor.is_alive():
            self.processor.join()

    def receive_image(self, msg):
        if msg != None and hasattr(msg, 'data'):
            self.get_logger().debug(F"FFinder: receive_image. dataLen={len(msg.data)}")
            self.image_queue.put(msg)

    def process_images(self):
        # Take images from the queue and find the faces therein
        while True:
            if self.processor_event.is_set():
                break
            try:
                msg = self.image_queue.get(block=True, timeout=2)
            except queue.Empty:
                msg = None
            if self.processor_event.is_set():
                break
            if msg != None:
                self.get_logger().debug('FFinder: process_image frame=%s, dataLen=%s'
                                    % (msg.header.frame_id, len(msg.data)) )
                # as of 20181016, the msg.data is returned as a list of ints. Convert to bytearray.
                converted_data = []
                with CodeTimer(self.get_logger().debug, 'convert to byte array'):
                    converted_data = bytearray(msg.data)

                # prepare the image and make suitable for face finding
                if self.param_image_compressed:
                    img = self.convert_image(converted_data)
                else:
                    img = converted_data

                # if there is a good image, find any faces
                if type(img) != type(None):
                    face_bounding_boxes = self.find_faces(img)
                    self.publish_bounding_boxes(face_bounding_boxes)

    def convert_image(self, raw_img):
        # convert the passed buffer into a proper Python image
        img = None
        try:
            # imageio.imread returns a numpy array where img[h][w] => [r, g, b]
            with CodeTimer(self.get_logger().debug, 'decompress image'):
                img = imageio.imread(io.BytesIO(raw_img))
            self.get_logger().debug('FFinder: imread image: h=%s, w=%s' % (len(img), len(img[0])))
        except Exception as e:
            self.get_logger().error('FFinder: exception uncompressing image. %s: %s'
                            % (type(e), e.args) )
            img = None
        return img

    def find_faces(self, img):
        # Given and image, find the faces therein and return the bounding boxes
        # Returns an array of 'dlib.rectangle'
        with CodeTimer(self.get_logger().debug, 'detect faces'):
            detected = self.detector(img, 0)
        self.get_logger().debug('FFinder: detected %s faces' % (len(detected)))
        for i, d in enumerate(detected):
            self.get_logger().debug("   face %s: Left: %s Top: %s Right: %s Bottom: %s" %
                        (i, d.left(), d.top(), d.right(), d.bottom()) )
        return detected

    def publish_bounding_boxes(self, bbs):
        # given a list of bounding boxes, publish same
        return

    def get_parameter_or(self, param, default):
        # Helper function to return value of a parameter or a default if not set
        ret = None
        param_desc = self.get_parameter(param)
        if param_desc.type_== Parameter.Type.NOT_SET:
            ret = default
        else:
            ret = param_desc.value
        return ret

class CodeTimer:
    # A little helper class for timing blocks of code
    def __init__(self, logger, name=None):
        self.logger = logger
        self.name = " '"  + name + "'" if name else ''

    def __enter__(self):
        self.start = time.clock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (time.clock() - self.start) * 1000.0
        self.logger('Code block' + self.name + ' took: ' + str(self.took) + ' ms')

def main(args=None):
    rclpy.init(args=args)

    ffNode = ROS2_facefinder_node()

    try:
        rclpy.spin(ffNode)
    except KeyboardInterrupt:
        ffNode.get_logger().info('FFinder: Keyboard interrupt')

    ffNode.stop_workers()

    ffNode.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()