from mmseg.apis import init_model, inference_model
import numpy as np
import cv2
from collections import namedtuple
import os
from tqdm import tqdm

cs_labels = namedtuple('cs_labels', ['name', 'id', 'color'])

cs_classes = [
    cs_labels('background', 0, (192, 192, 192)),  # Color: #C0C0C0
    cs_labels('headwear', 1, (245, 63, 63)),     # Color: #F53F3F
    cs_labels('hair', 2, (120, 22, 255)),        # Color: #7816FF
    cs_labels('gloves', 3, (0, 180, 42)),        # Color: #00B42A
    cs_labels('glasses', 4, (22, 93, 255)),      # Color: #165DFF
    cs_labels('coat', 5, (255, 125, 0)),         # Color: #FF7D00
    cs_labels('dress', 6, (235, 10, 164)),       # Color: #EB0AA4
    cs_labels('outerwear', 7, (123, 198, 22)),   # Color: #7BC616
    cs_labels('socks', 8, (134, 140, 156)),      # Color: #868C9C
    cs_labels('pants', 9, (219, 158, 0)),        # Color: #DB9E00
    cs_labels('skin', 10, (183, 29, 232)),       # Color: #B71DE8
    cs_labels('scarf', 11, (15, 198, 194)),      # Color: #0FC6C2
    cs_labels('skirt', 12, (255, 180, 0)),       # Color: #FFB400
    cs_labels('face', 13, (22, 141, 253)),       # Color: #168DFD
    cs_labels('shoes', 14, (255, 87, 34)),       # Color: #FF5722
    cs_labels('bag', 15, (255, 22, 162)),        # Color: #FF16A2
    cs_labels('accessory', 16, (22, 255, 185)),  # Color: #16FFB9
    cs_labels('jumpsuit', 17, (247, 17, 212)),   # Color: #F711D4
    # cs_labels('stand', 18, (212, 87, 106)),      # Color: #D4576A
    cs_labels('stand', 18, (192, 192, 192)),      # Color: #D4576A
]

config_path = 'configs/segformer/segformer_b5_768x1024_custom.py'
checkpoint_path = '/home/ubuntu/mmsegmentation/20000.pth'
images_dir = "/home/ubuntu/masked_validation_set/new_tryon"
output_dir = "/home/ubuntu/masked_validation_set/new_tryon-seg"
os.makedirs(output_dir, exist_ok=True)

images = os.listdir(images_dir)

model = init_model(config_path, checkpoint_path, device='cuda:0')

for image_file in tqdm(images, desc="Processing images"):
    img_path = os.path.join(images_dir, image_file)
    output_path = os.path.join(output_dir, image_file).replace(".jpg", ".png")
    result = inference_model(model, img_path)
    label = result.pred_sem_seg.data.squeeze().cpu().numpy()

    colored_mask = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cls in cs_classes:
        idx = (label == cls.id)
        if np.any(idx):
            colored_mask[idx] = (cls.color[2], cls.color[1], cls.color[0])

    cv2.imwrite(output_path, colored_mask)
