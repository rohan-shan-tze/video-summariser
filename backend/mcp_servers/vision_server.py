"""
Vision MCP Server.

Exposes two tools:
  detect_objects(video_path) -> {objects: [{label, confidence, frame_ts}]}
    Samples N frames from the mp4, runs YOLOv8n via OpenVINO IR,
    aggregates and deduplicates detected object labels.

  extract_text(video_path) -> {text_regions: [str], likely_has_graph: bool}
    Runs EasyOCR over sampled frames to extract visible text.
    Applies a keyword heuristic to guess whether a graph/chart is present.

Model:
  YOLOv8n converted to OpenVINO IR (FP16, ~6MB).
  Originally exported from ultralytics YOLOv8n COCO pretrained weights,
  then converted with `ovc`. Trained on COCO 80 classes.
  Stored in models/openvino/public/mobilenet-ssd/FP32/ (kept original path
  for consistency with download_models.py and .gitignore).

Device policy:
  OpenVINO device is "AUTO": uses Intel iGPU if available, falls back to CPU.
  NEVER "GPU" (Intel-only, not NVIDIA) and NEVER "cuda".
  Runs correctly on any CPU-only reviewer machine.

Run as a subprocess; communicates via stdio JSON-RPC (see mcp_common/server.py).
"""

import sys
import io
from pathlib import Path
import numpy as np

# EasyOCR prints Unicode progress chars during model download.
# Force UTF-8 on Windows consoles to avoid UnicodeEncodeError.
if hasattr(sys.stderr, "buffer") and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.mcp_common.server import MCPServer


# Constants

# Number of frames to sample per video. 8 is enough for ~1-minute clips.
_N_FRAMES = 8

# IR model location (populated by download_models.py / setup process).
_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "openvino"
_MODEL_XML  = _MODEL_DIR / "public" / "mobilenet-ssd" / "FP32" / "mobilenet-ssd.xml"

# COCO 80-class labels (YOLOv8 class order).
_COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Confidence threshold: detections below this are ignored.
_CONFIDENCE_THRESHOLD = 0.35

# Keywords that suggest a graph or chart is present in OCR-extracted text.
_GRAPH_KEYWORDS = {
    "axis", "axes", "graph", "chart", "plot", "figure", "fig",
    "table", "legend", "xlabel", "ylabel", "x-axis", "y-axis",
    "%", "mean", "median", "correlation", "distribution", "histogram",
    "bar", "pie", "scatter", "regression", "p-value", "n=",
}

# Lazy-loaded singletons

_ov_compiled = None
_ocr_reader  = None


def _get_ov_model():
    """Load and compile the OpenVINO IR model once, cache it."""
    global _ov_compiled
    if _ov_compiled is not None:
        return _ov_compiled

    if not _MODEL_XML.exists():
        raise FileNotFoundError(
            f"OpenVINO model not found at {_MODEL_XML}. "
            "Run the model setup step (see docs/SETUP.md)."
        )

    import openvino as ov
    core  = ov.Core()
    model = core.read_model(str(_MODEL_XML))
    # device="AUTO": OpenVINO picks the best available device.
    # On Intel machines this may use the iGPU; everywhere else it uses CPU.
    # This NEVER touches an NVIDIA GPU, OpenVINO has no CUDA backend.
    _ov_compiled = core.compile_model(model, device_name="AUTO")
    return _ov_compiled


def _get_ocr_reader():
    """Load EasyOCR once and cache it."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        easyocr_dir = Path(__file__).parent.parent.parent / "models" / "easyocr"
        easyocr_dir.mkdir(exist_ok=True)
        # gpu=False: CPU inference only, never use CUDA.
        _ocr_reader = easyocr.Reader(
            ["en"],
            model_storage_directory=str(easyocr_dir),
            gpu=False,
        )
    return _ocr_reader



# Frame sampling

def _sample_frames(video_path: str, n: int = _N_FRAMES) -> list:
    """
    Return a list of (frame_bgr, timestamp_seconds) tuples, evenly spaced
    across the video. Uses OpenCV, no ffmpeg needed for frame extraction.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    indices = [int(i * total / n) for i in range(n)]
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            ts = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)
            frames.append((frame, ts))
    cap.release()
    return frames


# YOLOv8 output parsing

def _parse_yolov8_output(output: "np.ndarray", conf_threshold: float) -> list:
    """
    Parse YOLOv8 output tensor [1, 84, 8400] into detection dicts.

    YOLOv8 layout per anchor: [cx, cy, w, h, class0_score, ..., class79_score]
    We take the argmax class score as the predicted class, use it as confidence.

    Returns a list of {class_id, confidence} dicts above the threshold.
    """
    import numpy as np
    # Transpose to [8400, 84] so each row is one anchor's predictions.
    preds = output[0].T        # shape: (8400, 84)
    class_scores = preds[:, 4:]  # shape: (8400, 80) — drop the 4 box coords

    # For each anchor, find the class with the highest score.
    class_ids   = np.argmax(class_scores, axis=1)   # (8400,)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]  # (8400,)

    # Filter by threshold.
    mask = confidences >= conf_threshold
    return [
        {"class_id": int(class_ids[i]), "confidence": float(confidences[i])}
        for i in np.where(mask)[0]
    ]


# Tool: detect_objects

def detect_objects(video_path: str) -> dict:
    """
    Sample N frames, run YOLOv8n via OpenVINO on each, return the unique
    detected object labels with their highest confidence and timestamp.

    Returns:
      objects - list of {label, confidence, frame_ts}, one entry per unique
                label (highest-confidence instance kept).
    """
    import numpy as np, cv2

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    compiled    = _get_ov_model()
    input_layer = compiled.input(0)   # expects [1, 3, 640, 640] float32
    H, W        = 640, 640

    frames = _sample_frames(video_path)
    # best: label -> {label, confidence, frame_ts}
    best: dict[str, dict] = {}

    for frame_bgr, ts in frames:
        # Preprocess: resize, BGR->RGB, HWC->CHW, normalise to [0,1], add batch dim.
        resized = cv2.resize(frame_bgr, (W, H))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob    = (rgb.transpose(2, 0, 1)[np.newaxis] / 255.0).astype(np.float32)

        output  = compiled({input_layer: blob})[compiled.output(0)]
        detections = _parse_yolov8_output(output, _CONFIDENCE_THRESHOLD)

        for det in detections:
            cid   = det["class_id"]
            conf  = det["confidence"]
            label = _COCO_LABELS[cid] if cid < len(_COCO_LABELS) else f"class_{cid}"
            if label not in best or conf > best[label]["confidence"]:
                best[label] = {
                    "label":      label,
                    "confidence": round(conf, 3),
                    "frame_ts":   ts,
                }

    return {"objects": list(best.values())}


# Tool: extract_text

def extract_text(video_path: str) -> dict:
    """
    Run EasyOCR over sampled frames. Return extracted text regions and a
    heuristic bool for graph/chart presence.

    Graph heuristic: if any OCR word matches a known graph/chart keyword
    (axis labels, statistical terms, etc.) we flag likely_has_graph=True.
    This is intentionally simple, good enough to satisfy the spec query
    "are there any graphs in the video?"

    Returns:
      text_regions     deduplicated list of text strings found across frames
      likely_has_graph True if graph-related keywords detected
    """
    import cv2

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader     = _get_ocr_reader()
    frames     = _sample_frames(video_path)
    seen_texts: set[str] = set()
    likely_has_graph     = False

    for frame_bgr, _ts in frames:
        # EasyOCR expects RGB; OpenCV gives BGR.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        texts     = reader.readtext(frame_rgb, detail=0)

        for t in texts:
            t = t.strip()
            if not t:
                continue
            seen_texts.add(t)
            words = {w.lower().rstrip(".,;:") for w in t.split()}
            if words & _GRAPH_KEYWORDS:
                likely_has_graph = True

    return {
        "text_regions":     sorted(seen_texts),
        "likely_has_graph": likely_has_graph,
    }


# Entrypoint

if __name__ == "__main__":
    server = MCPServer("vision_server")
    server.register_tool("detect_objects", detect_objects)
    server.register_tool("extract_text",   extract_text)
    server.run()
