import json
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint
from imgaug.augmentables import KeypointsOnImage


def get_points(json_info):
    points = []
    if isinstance(json_info, dict):
        for key, value in json_info.items():
            if key == 'points':
                points += value
            else:
                points += get_points(value)
    elif isinstance(json_info, list):
        for element in json_info:
            points += get_points(element)

    return points


def set_points(json_info, points):
    if isinstance(json_info, dict):
        for key, value in json_info.items():
            if key == 'points':
                for i in range(len(value)):
                    value[i] = points.pop(0)
            else:
                set_points(value, points)

    elif isinstance(json_info, list):
        for element in json_info:
            set_points(element, points)

    return json_info


class Augmenter():
    def apply(self, image, label, augmenter: iaa):
        # get all points from label
        points, label = self.get_points(label)
        points = [Keypoint(x=point[0], y=point[1]) for point in points]
        points = KeypointsOnImage(keypoints=points, shape=image.shape)

        # apply augmenter to image and label
        image, points = augmenter(image=image, keypoints=points)

        # set all transformed points to new label (json info)
        points = [[float(point.x), float(point.y)] for point in points.keypoints]
        label = self.set_points(label, points)
        label['imageHeight'], label['imageWidth'] = image.shape[0], image.shape[1]

        return image, label


    def get_points(self, label):
        if isinstance(label, str):
            with open(file=label, mode='r', encoding='utf-8') as f:
                label = json.load(f)
            points = get_points(label)
        elif isinstance(label, dict):
            points = get_points(label)
        else:
            raise TypeError('label must be str, dict.')
        return points, label

    def set_points(self, label, points):
        label = set_points(label, points)
        return label


if __name__ == '__main__':
    import cv2
    import numpy as np

    augmenter = Augmenter()

    image_path = '...'
    label_path = '...json'

    image = cv2.imread(image_path)
    thickness = max(image.shape) // 400

    with open(file=label_path, mode='r', encoding='utf-8') as f:
        label = json.load(f)

    image, label = augmenter.apply(
        image=image,
        label=label,
        augmenter=iaa.Rotate(rotate=(-90, 90), fit_output=True)
    )

    for shape in label['shapes']:
        if shape['shape_type'] == 'rectangle':
            points = np.int32(shape['points'])
            cv2.rectangle(img=image, pt1=tuple(points[0]), pt2=tuple(points[1]), color=(0, 255, 0), thickness=thickness)
        elif shape['shape_type'] == 'polygon':
            cv2.polylines(img=image, pts=[np.int32(shape['points'])], isClosed=True, color=(0, 255, 0), thickness=thickness)
        else:
            raise ValueError(f"visual function for {shape['shape_type']} is not implemented.")

    cv2.imwrite('image.png', image)
