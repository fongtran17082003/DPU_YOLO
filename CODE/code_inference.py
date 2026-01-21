import onnxruntime as ort
import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
import os

# ===============================
# ARGPARSE
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 ONNX Video Inference")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save frames"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Input image size (h w)"
    )

    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")

    return parser.parse_args()


# ===============================
# LETTERBOX
# ===============================
def letterbox(img, new_shape=(224, 224), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)

    new_w, new_h = int(round(w * r)), int(round(h * r))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = new_shape[1] - new_w
    dh = new_shape[0] - new_h
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img, r, (dw, dh)


# ===============================
# POSTPROCESS YOLOv11
# ===============================
def postprocess_output(output, img_shape, ratio, pad,
                       conf_threshold=0.4, iou_threshold=0.45):

    output = output[0].transpose()  # [N, C]

    boxes = output[:, :4]           # cx, cy, w, h
    scores = output[:, 4:]          # class scores

    class_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    mask = class_scores >= conf_threshold
    boxes = boxes[mask]
    class_scores = class_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes /= ratio

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_shape[0])

    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        class_scores.tolist(),
        conf_threshold,
        iou_threshold
    )

    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])

    indices = indices.flatten()
    return boxes[indices], class_scores[indices], class_ids[indices]


# ===============================
# DRAW BOXES
# ===============================
COCO_CLASSES = ["person"]

def draw_boxes(img, boxes, scores, class_ids):
    img = img.copy()
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        label = f"{COCO_CLASSES[cid]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    return img


# ===============================
# MAIN
# ===============================
def main():
    args = parse_args()

    model_path = args.model
    video_path = args.video
    output_dir = Path(args.output)
    img_size = tuple(args.img_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== YOLOv11 ONNX VIDEO INFERENCE =====")
    print("Model :", model_path)
    print("Video :", video_path)
    print("Output:", output_dir)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "❌ Cannot open video"

    frame_id = 0
    total_pre = total_inf = total_post = 0.0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Preprocess
        t0 = time.time()
        img_lb, ratio, pad = letterbox(frame, img_size)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)[None]
        t1 = time.time()

        # Inference
        output = session.run(
            [output_name],
            {input_name: img_input}
        )[0]
        t2 = time.time()

        # Postprocess
        boxes, scores, class_ids = postprocess_output(
            output, frame.shape, ratio, pad, args.conf, args.iou
        )
        t3 = time.time()

        # Save
        frame_draw = draw_boxes(frame, boxes, scores, class_ids)
        cv2.imwrite(str(output_dir / f"frame_{frame_id:06d}.jpg"), frame_draw)

        total_pre += (t1 - t0)
        total_inf += (t2 - t1)
        total_post += (t3 - t2)

    cap.release()
    total_time = time.time() - start_time

    print("\n========== SUMMARY ==========")
    print(f"Frames         : {frame_id}")
    print(f"Total time     : {total_time:.2f} s")
    print(f"FPS            : {frame_id / total_time:.2f}")
    print(f"Preprocess avg : {total_pre / frame_id * 1000:.2f} ms")
    print(f"Inference avg  : {total_inf / frame_id * 1000:.2f} ms")
    print(f"Postprocess avg: {total_post / frame_id * 1000:.2f} ms")
    print("=============================")


if __name__ == "__main__":
    main()
