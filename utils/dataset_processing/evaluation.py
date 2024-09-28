import numpy as np
import matplotlib.pyplot as plt

from .grasp import GraspRectangles, detect_grasps


def plot_output(didx, rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """

    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    plot = ax.imshow(grasp_q_img, cmap='cubehelix', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = plt.subplot(2, 2, 3)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)

    ax = plt.subplot(2, 2, 4)
    plot = ax.imshow(grasp_width_img, cmap='jet', vmin=0, vmax=100)
    ax.set_title('Width')

    ax.axis('off')
    plt.colorbar(plot)
    plt.savefig("/home/meiguiz/Downloads/gradet-main/results/wuiou/9/{}.png".format(didx), dpi=600)



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(2, 2, 1)
    # ax.imshow(rgb_img)
    # for g in gs:
    #     g.plot(ax)
    # ax.set_title('RGB')
    # ax.axis('off')
    #
    # ax = fig.add_subplot(2, 2, 2)
    # ax.imshow(depth_img, cmap='gray')
    # for g in gs:
    #     g.plot(ax)
    # ax.set_title('Depth')
    # ax.axis('off')
    #
    # ax = fig.add_subplot(2, 2, 3)
    # plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    # ax.set_title('Q')
    # ax.axis('off')
    # plt.colorbar(plot)
    #
    # ax = fig.add_subplot(2, 2, 4)
    # plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    # ax.set_title('Angle')
    # ax.axis('off')
    # plt.colorbar(plot)
    # plt.show()

def plot_label_rectangle(rgb_img, GraspRectangles, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):

    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)


    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(rgb_img)
    index = 0
    # for g in GraspRectangles:
    #     # if index % 4 in [0]:
    #     g.plot(ax)
    #     index += 1     #remove some ground truth to see clearly
    # for g in gs:
    #     g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    plt.show()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs     #a set of ground truth bounding box
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)   #predict the image
    for g in gs:
        if g.max_giou(gt_bbs) > 0.25:
            print(g.max_iou(gt_bbs))
            return True
    else:
        return False


def calculate_GIOU_predata(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    the ability of the method is compute the precondition of GIOU loss
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs     #a set of ground truth bounding box
    #if training, there is one grasp
    g = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)   #predict the image
    if len(g) == 0:
        return 0, 0.5, 1
    max_iou, union, enclose_area = g[0].max_iou(gt_bbs)
    return max_iou, union, enclose_area
