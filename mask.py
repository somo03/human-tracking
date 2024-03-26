import cv2
import numpy as np


def mask_frame(frame, bbox, blur=False, blur_strength=99, bbox_expansion=0.10):
    """
    Takes a frame read by opencv, start- and end-point coordinates of bounding box and hides everything outside the
    bounding box.
    If 'blur', then hiding means blurring. Otherwise, the area outside bbox is blacked out.
    'bbox_expansion' regulates the degree the bounding box from human tracking software is expanded.
    This function operates inplace (modifies memory where 'frame' argument is stored).
    """

    height, width, nchannels = frame.shape
    mask = np.zeros((height, width, nchannels), dtype=np.uint8)
    xd, yd, xu, yu = bbox

    # expand bbox a bit to avoid overshadowing edges of person walking
    bbox_height = yu - yd
    bbox_width = xu - xd
    if (xd - bbox_width * bbox_expansion) >= 0: xd = int(xd - bbox_width * bbox_expansion)
    if (yd - bbox_height * bbox_expansion) >= 0: yd = int(yd - bbox_height * bbox_expansion)
    if (xu + bbox_width * bbox_expansion) <= width: xu = int(xu + bbox_width * bbox_expansion)
    if (yu + bbox_height * bbox_expansion) <= height: yu = int(yu + bbox_height * bbox_expansion)

    mask[yd:yu, xd:xu, :] = 255
    # invert mask
    mask = cv2.bitwise_not(mask)

    if blur:
        blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        frame[mask > 0] = blurred[mask > 0]
    else:
        frame[mask > 0] = 0


path = "C:\\Users\\somo03\\PythonProjects\\OpenPose_quick_start\\openpose_quick_start\\openpose\\examples\\media\\COCO_val2014_000000000192.jpg"
frame_test = cv2.imread(path)

bbox = [0, 0, 300, 300]
blur_strength = 25

if __name__ == "__main__":

    # test on some image
    mask_frame(frame_test, bbox=bbox)
    cv2.imshow("image", frame_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
