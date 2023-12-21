from human_tracking_api import TrackingAPI

tracker = TrackingAPI(
    1,
    "C:/Users/somo03/PythonProjects/data/video/walking_only/B34_Xsens/Runde 4/Matte.mp4",
    "./data/output/B34_R4_Matte_blur_expanded.mp4"
)
tracker.track_people()
bboxes = tracker.get_bb_list()

with open(r'./tests/output.txt', 'w') as fp:
    for item in bboxes:
        fp.write("%s\n" % item)
