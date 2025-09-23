import cv2


# Method: Used to calculate the ratio between intersection and union of 2 bounding boxes
def box_iou_ratio(a, b):
    """
    :param a: Box A
    :param b: Box B
    :return: Ratio = AnB / AuB
    """
    w_intersection = max(0, (min(a[2], b[2]) - max(a[0], b[0])))
    h_intersection = max(0, (min(a[3], b[3]) - max(a[1], b[1])))
    s_intersection = w_intersection * h_intersection

    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intersection) / (s_a + s_b - s_intersection)


# Method: Used to draw the bounding boxes and labels on an image
def draw_box_label(image, box, colour=(0, 255, 255)):
    """
    :param image: Image
    :param box: Bounding Box
    :param colour: Colour for the Bounding Box
    :return: Image with bounding box (and label)
    """
    left, top, right, bottom = box[1], box[0], box[3], box[2]

    # Draw the bounding box
    cv2.rectangle(image, (left, top), (right, bottom), colour, 4)

    return image

