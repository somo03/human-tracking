import argparse
from os import path
from time import perf_counter

import cv2
import numpy as np

from api.deepsort import DeepSORTTracker
from api.yolo import YOLOPersonDetector

# constants
YOLO_MODEL = "./checkpoints/yolov7x.pt"
REID_MODEL = "./checkpoints/ReID.pb"
MAX_COS_DIST = 0.5
MAX_TRACK_AGE = 100


class TrackingAPI:
    def __init__(self, trackid, input_vid):
        self.trackid = trackid
        self.input_vid = input_vid
        self.bb_list = []

    def get_trackid(self):
        return self.trackid

    def get_bb_list(self):
        return self.bb_list

    def track_people(self):

        # initialize Yolo person detector and DeepSORT tracker
        detector = YOLOPersonDetector()
        detector.load(YOLO_MODEL)
        tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE)

        # initialize video stream objects
        video = cv2.VideoCapture(self.input_vid)

        # core processing loop
        frame_i = 0
        time_taken = 0
        while True:
            start = perf_counter()

            # read input video frame
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # process YOLO detections
            detections = detector.detect(frame)
            try:
                bboxes, scores, _ = np.hsplit(detections, [4, 5])
                bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
                n_objects = detections.shape[0]
            except ValueError:
                bboxes = np.empty(0)
                scores = np.empty(0)
                n_objects = 0

            # track targets by refining with DeepSORT
            bbox = tracker.track(frame, bboxes, scores.flatten(), bbox_by_id_only=True, trackid=self.trackid)
            self.bb_list.append(bbox)

            # calculate FPS and display output frame
            frame_time = perf_counter() - start
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("<< User has terminated the process >>")
                break
            time_taken += frame_time
            frame_i += 1
            print(
                f"Frame {frame_i}: "
                f"{n_objects} people - {int(frame_time*1000)} ms = {1/frame_time:.2f} Hz"
            )

        # print performance metrics
        print(
            f"\nTotal frames processed: {frame_i}"
            f"\nVideo processing time: {time_taken:.2f} s"
            f"\nAverage FPS: {frame_i/time_taken:.2f} Hz"
        )


