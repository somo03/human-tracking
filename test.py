from human_tracking_api import TrackingAPI
import os


#INPUT_VID = "C:\\Users\\somo03\\PythonProjects\\data\\video\\full\\B34_Xsens\\Moticon\\Runde 1\\Matte.mp4"
INPUT_VID = "C:\\Users\\somo03\\PythonProjects\\data\\video\\full\\B15_Xsens\\Moticon\\Runde 3\\matte\\Matte.mp4"
print(os.getcwd())
tracker = TrackingAPI(
    2,
    INPUT_VID,
    "./data/output/B15_R3_Matte_test.mp4"
)
tracker.track_people()
#bboxes = tracker.get_bb_liqst()

'''
with open(r'./tests/output.txt', 'w') as fp:
    for item in bboxes:
        fp.write("%s\n" % item)
'''