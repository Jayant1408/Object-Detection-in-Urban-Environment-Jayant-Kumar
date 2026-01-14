import os, glob
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Udacity visualization helper
import visualization_utils as viz

from object_detection.utils import label_map_util

MODEL_DIR = os.environ["MODEL_DIR"]
LABEL_MAP = os.environ["LABEL_MAP"]
TEST_IMG_DIR = os.environ["TEST_IMG_DIR"]
OUT_DIR = os.environ.get("OUT_DIR", "inference_output")
VIDEO_NAME = os.environ.get("VIDEO_NAME", "output.mp4")

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...")
detect_fn = tf.saved_model.load(MODEL_DIR)
print("Model loaded")

category_index = label_map_util.create_category_index_from_labelmap(
    LABEL_MAP, use_display_name=True
)

img_paths = sorted(
    glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")) +
    glob.glob(os.path.join(TEST_IMG_DIR, "*.jpeg")) +
    glob.glob(os.path.join(TEST_IMG_DIR, "*.png"))
)

if not img_paths:
    raise SystemExit(f"No images found in {TEST_IMG_DIR}")

# Read first image to get size
first_img = np.array(Image.open(img_paths[0]).convert("RGB"))
height, width, _ = first_img.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_path = os.path.join(OUT_DIR, VIDEO_NAME)
writer = cv2.VideoWriter(video_path, fourcc, 10, (width, height))

for idx, p in enumerate(img_paths):
    image = Image.open(p).convert("RGB")
    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    num = int(detections["num_detections"][0])
    boxes = detections["detection_boxes"][0][:num].numpy()
    classes = detections["detection_classes"][0][:num].numpy().astype(np.int32)
    scores = detections["detection_scores"][0][:num].numpy()

    viz.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.30,
        line_thickness=4
    )

    # Save image
    out_img_path = os.path.join(OUT_DIR, os.path.basename(p))
    Image.fromarray(image_np).save(out_img_path)

    # Write frame to video
    frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)

    if idx % 10 == 0:
        print(f"Processed frame {idx}")

writer.release()
print(f"[DONE] Images + video written to {OUT_DIR}")
print(f"[VIDEO] {video_path}")
