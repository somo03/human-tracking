import cv2
import matplotlib.pyplot as plt
import numpy as np

from deepsort.detection import Detection
from deepsort.generate_detections import create_box_encoder
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.tracker import Tracker

from blur import blur_frame


class DeepSORTTracker:
    """
    DeepSORT tracker wrapper, exposing just one method which receives an image and its detections to
    refine them using existing tracks
    """

    def __init__(
        self,
        reid_model: str,
        cosine_thresh: float,
        max_track_age: int,
        nn_budget: int = None,
    ) -> None:
        self.encoder = create_box_encoder(reid_model, batch_size=1)
        self.tracker = Tracker(
            NearestNeighborDistanceMetric("cosine", cosine_thresh, nn_budget),
            max_age=max_track_age,
        )
        self.colors = plt.get_cmap("hsv")(np.linspace(0, 1, 20, False))[:, :3] * 255

    def track(self, frame: np.ndarray, bboxes: np.ndarray, scores: np.ndarray, bbox_by_id_only=False, trackid: int = None) -> None:
        """
        Accepts an image and its YOLO detections, uses these detections and existing tracks to get a
        final set of bounding boxes, which are then drawn onto the input image if bbox_by_id_only == True

        Parameters
        bbox_by_id_only : bool
            True if function is used to track one particular object with id equal to 'trackid'. In this case bbox is NOT
            drawn on the frame.
            False if function is used to track all objects present in the frame. In this case detected bboxes are drawn
            on the frame.

        Returns
        Return bounding box coordinates of object with id equal to 'trackid' only if bbox_by_id_only == True, otherwise
        doesn't return anything.
        """
        feats = self.encoder(frame, bboxes)
        dets = [Detection(*args) for args in zip(bboxes, scores, feats)]

        # refine the detections
        self.tracker.predict()
        self.tracker.update(dets)

        if bbox_by_id_only and (trackid is None):
            raise TypeError("Argument 'trackid' must be an integer if 'bbox_by_id_only' is set to True")

        # render the final tracked bounding boxes on the input frame
        # loop goes through the list of all tracked objects and draws bounding boxes
        # we need just one id (track.track_id) from the list corresponding to id of a person with insoles
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr().astype(np.int32)  # extract bbox coordinates here
            # bbox is a list of 4 numbers: x and y coordinates of start- and end-point of the bbox
            # if trackid object is found, blur frame outside bbox, return its bbox coordinates, and exit the loop
            if bbox_by_id_only and (trackid == track.track_id):
                blur_frame(frame, bbox)
                return bbox

            # bbox_by_id_only regulates whether to draw bbox in the frame
            if not bbox_by_id_only:
                # draw detection bounding box
                color = self.colors[track.track_id % 20]
                cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                # draw text box for printing ID
                cv2.rectangle(
                    frame,
                    tuple(bbox[:2]),
                    (bbox[0] + (4 + len(str(track.track_id))) * 8, bbox[1] + 20),
                    color,
                    -1,
                )
                # print ID in the text box
                cv2.putText(
                    frame,
                    f"ID: {track.track_id}",
                    (bbox[0] + 4, bbox[1] + 13),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.4,
                    (0, 0, 0),
                    lineType=cv2.LINE_AA,
                )

        # if 'trackid' object is not found in the frame, return None
        if bbox_by_id_only:
            return None

