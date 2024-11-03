import argparse
import os
import cv2
import torch
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
# 1280,720
# 1920,1200
# 600, 375
def load_and_resize_image(image_path, target_size=(1280, 720)):
    """Load an image and resize it to the target size."""
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img

def save_segmentation_result(model, img, output_path, palette, opacity=0.5):
    """Perform segmentation on the image and save the result."""
    result = inference_segmentor(model, img)
    img_result = model.show_result(img, result, palette=palette, show=False, opacity=opacity)
    cv2.imwrite(output_path, img_result)
    print(f"Saved segmentation result to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run segmentation inference and save results')
    parser.add_argument('config', nargs='?', default='./configs/ours/ganav_group6_lake.py', help='Path to config file')
    parser.add_argument('checkpoint', nargs='?', default='./work_dirs/ganav_group6_lake/latest.pth', help='Path to checkpoint file')
    parser.add_argument('image', nargs='?', default='./data/lake/image', help='Path to input image or directory of images')
    parser.add_argument('output', nargs='?', default='./results', help='Path to save the output image(s)')
    parser.add_argument('--device', default='cuda:0', help='Device for model inference')
    parser.add_argument('--palette', default='lake_group', help='Color palette used for segmentation map')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of segmentation overlay')
    parser.add_argument('--resize', type=tuple, default=(1280, 720), help='Resize dimensions for input images (width, height)')
    args = parser.parse_args()

    # Initialize segmentation model
    palette = get_palette(args.palette)
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # Check if the input is a directory or a single image
    if os.path.isdir(args.image):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.image):
            file_path = os.path.join(args.image, filename)
            if os.path.isfile(file_path):
                img = load_and_resize_image(file_path, target_size=args.resize)
                output_path = os.path.join(args.output, f'result_{filename}')
                save_segmentation_result(model, img, output_path, palette, opacity=args.opacity)
    else:
        img = load_and_resize_image(args.image, target_size=args.resize)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_segmentation_result(model, img, args.output, palette, opacity=args.opacity)

if __name__ == '__main__':
    main()
