import onnxruntime as ort
import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# ===============================
# ARGPARSE
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11 ONNX Image Inference (Vitis AI DPU)")

    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--input", type=str, required=True, help="Path to input image or folder")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224], help="Input size (h w)")
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

    dw = (new_shape[1] - new_w) / 2
    dh = (new_shape[0] - new_h) / 2

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
    boxes = output[:, :4]
    scores = output[:, 4:]

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
        cv2.putText(
            img, label, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return img


# ===============================
# CREATE VITIS AI SESSION
# ===============================
def create_vitisai_session(model_path: str, vaip_config: str):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 2

    try:
        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": vaip_config}],
            disable_fallback=True,
        )
        print("✅ Vitis AI DPU session created (no fallback)")
    except TypeError:
        sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["VitisAIExecutionProvider"],
            provider_options=[{"config_file": vaip_config}],
        )
        print("⚠️ Vitis AI DPU session created (fallback possible)")

    print("Execution providers:", sess.get_providers())
    return sess


# ===============================
# MAIN
# ===============================
def main():
    args = parse_args()

    model_path = args.model
    input_path = Path(args.input)
    output_dir = Path(args.output)
    img_size = tuple(args.img_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    if input_path.is_dir():
        image_list = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        )
    else:
        image_list = [input_path]

    assert len(image_list) > 0, "❌ No input images found"

    print("===== YOLOv11 ONNX IMAGE INFERENCE (DPU) =====")
    print("Model :", model_path)
    print("Input :", input_path)
    print("Images:", len(image_list))
    print("Output:", output_dir)

    # Create DPU session
    vaip_config = "/usr/bin/vaip_config.json"
    session = create_vitisai_session(model_path, vaip_config)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    total_pre = total_inf = total_post = 0.0
    start_time = time.time()

    # Optional warmup
    dummy = np.zeros((1, 3, img_size[0], img_size[1]), dtype=np.float32)
    for _ in range(3):
        session.run([output_name], {input_name: dummy})

    for img_path in image_list:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"❌ Cannot read image: {img_path}")
            continue

        t0 = time.time()
        img_lb, ratio, pad = letterbox(frame, img_size)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_input = img_rgb.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)[None]
        t1 = time.time()

        output = session.run([output_name], {input_name: img_input})[0]
        t2 = time.time()

        boxes, scores, class_ids = postprocess_output(
            output, frame.shape, ratio, pad, args.conf, args.iou
        )
        t3 = time.time()

        frame_draw = draw_boxes(frame, boxes, scores, class_ids)
        cv2.imwrite(str(output_dir / img_path.name), frame_draw)

        total_pre += (t1 - t0)
        total_inf += (t2 - t1)
        total_post += (t3 - t2)

    total_time = time.time() - start_time
    n = len(image_list)

    print("\n========== SUMMARY ==========")
    print(f"Images         : {n}")
    print(f"Total time     : {total_time:.2f} s")
    print(f"FPS            : {n / total_time:.2f}")
    print(f"Preprocess avg : {total_pre / n * 1000:.2f} ms")
    print(f"Inference avg  : {total_inf / n * 1000:.2f} ms")
    print(f"Postprocess avg: {total_post / n * 1000:.2f} ms")
    print("=============================")


if __name__ == "__main__":
    main()
