from skimage.draw import polygon
import numpy as np
import cv2
import glob
import os
#
# file_path = "data"
# graspf = glob.glob(os.path.join(file_path, '*', 'pcd*cpos.txt'))
#
# depthf = [f.replace('cpos.txt', 'd.tiff') for f in graspf]
# rgbf = [f.replace('d.tiff', 'r.png') for f in depthf]
# print(graspf[:10])
# print(depthf[:10])
# print(rgbf[:10])
# def iou(gr1, gr2):
#     """
#     Compute IoU with another grasping rectangle
#     :param gr: GraspingRectangle to compare
#     :param angle_threshold: Maximum angle difference between GraspRectangles
#     :return: IoU between Grasp Rectangles
#     """
#
#     rr1, cc1 = polygon(gr1[:, 0], gr1[:, 1])
#     rr2, cc2 = polygon(gr2[:, 0], gr2[:, 1])
#
#
#     try:
#         r_min = min(rr1.min(), rr2.min())
#         c_min = min(cc1.min(), cc2.min())
#         r_max = max(rr1.max(), rr2.max()) + 1
#         print("r_max", r_max)
#         c_max = max(cc1.max(), cc2.max()) + 1
#         print("c_max", c_max)
#     except:
#         return 0
#
#     canvas = np.zeros((r_max-r_min, c_max-c_min))
#     enclose_area = canvas.shape[0] * canvas.shape[1]
#     print("enclose_area", enclose_area)
#     canvas[rr1-r_min, cc1-c_min] += 1
#     canvas[rr2-r_min, cc2-c_min] += 1
#     cv2.imshow('img', canvas)
#     cv2.waitKey(0)
#     union = np.sum(canvas > 0)
#     if union == 0:
#         return 0
#     intersection = np.sum(canvas == 2)
#     return intersection / union, union, enclose_area
#
#
# grasp_bounding_box1 = np.array([[50, 50], [50, 175], [175, 175], [175, 50]])
# grasp_bounding_box2 = np.array([[100, 100], [150, 150], [100, 200], [200, 200], [150, 150], [200, 100]])
# iou_res, union, enclose_area = iou(grasp_bounding_box1, grasp_bounding_box2)
# res = iou_res - (enclose_area - union) / enclose_area
# print(res)