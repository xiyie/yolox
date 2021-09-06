import logging as log
import cv2

import os
import json
from exp import get_exp
import torch
import numpy as np
from utils import postprocess, convert_to_voc_format, preproc

log.basicConfig(level=log.DEBUG)

label_id_map = {0: "face_mask", 1: "face"}


def init():
    """Initialize model
    Returns: model
    """
    # args = argparse.ArgumentParser()
    exp = get_exp('yolox_voc_tiny.py', None)  #
    # exp.merge([])
    device = "cuda:0"
    model = exp.get_model()
    model.to(device)

    model_path = "/usr/local/ev_sdk/model/latest_ckpt.pth"  #
    #     model_path =  "/usr/local/ev_sdk/model/last_mosaic_epoch_ckpt.pth"  #
    if not os.path.isfile(model_path):
        log.error(f'{model_path} does not exist')
        return None
    log.info('Loading model...')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    # state_dict = torch.jit.load(model_path)

    model = model.eval()

    return model


def process_image(net, input_image, args=None):
    num_classes = len(label_id_map)
    with torch.no_grad():
        # image_ = Image.fromarray(image)
        # image = image_.convert("RGB")
        # image_shape = np.array(np.shape(image)[0:2])
        # images = [input_image]
        #
        # images = torch.from_numpy(np.asarray(images))
        h, w, d = input_image.shape
        info_imgs = [torch.tensor([h]), torch.tensor([w])]
        img, _ = preproc(input_image, (416, 416))

        images = [img]
        img = torch.from_numpy(np.asarray(images)).to('cuda:0')
        data_list = {}
        outputs = net(img)
        outputs = postprocess(outputs, num_classes, 0.1, 0.65)

        output_list = []
        # for i in range(3):
        #     output_list.append(yolo_decodes[i](outputs[i]))
        data_list.update(convert_to_voc_format(outputs, info_imgs, ids=torch.tensor([0])))
        # all_boxes = [
        #     [[] for _ in range(1)] for _ in range2222(num_classes)
        # ]

        detect_objs = []
        for img_num in range(1):
            bboxes, cls, scores = data_list[img_num]
            if bboxes is None:
                continue
            # if bboxes is None:
            #     for j in range(num_classes):
            #         all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
            #     continue
            # for j in range(num_classes):
            #     mask_c = cls == j
            #     if sum(mask_c) == 0:
            #         all_boxes[j][img_num] = np.empty([0, 5], dtype=np.float32)
            #         continue
            for i in range(cls.shape[0]):
                detect_objs.append({
                    'xmin': int(bboxes[i][0]),
                    'ymin': int(bboxes[i][1]),
                    'xmax': int(bboxes[i][2]),
                    'ymax': int(bboxes[i][3]),
                    'confidence': scores[i].tolist(),
                    'name': label_id_map[cls[i].tolist()],

                })

    return json.dumps({"objects": detect_objs})


if __name__ == '__main__':
    """Test python api
    """

    img = cv2.imread('/home/data/52/4.jpg')  #

    predictor = init()

    result = process_image(predictor, img)

    log.info(result)