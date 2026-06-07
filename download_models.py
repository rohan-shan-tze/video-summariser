"""
Download all required models. Run once; after this the app is fully offline.
Usage: python download_models.py
"""
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download_whisper():
    print("[whisper] Downloading faster-whisper base model (int8, ~150 MB)...")
    from faster_whisper import WhisperModel
    whisper_dir = MODELS_DIR / "whisper"
    whisper_dir.mkdir(exist_ok=True)
    # device="cpu" and compute_type="int8" are the committed defaults — GPU-free.
    # download_root forces the cache into our models/ dir rather than ~/.cache
    WhisperModel("base", device="cpu", compute_type="int8",
                 download_root=str(whisper_dir))
    print("[whisper] Done.")


def download_openvino_model():
    """
    Download YOLOv8n COCO weights and convert to OpenVINO IR.

    Why YOLOv8n instead of the original plan (MobileNet-SSD via omz_downloader)?
      - openvino-dev (which ships omz_downloader) does not install on Python 3.13.
      - The OpenVINO storage server no longer serves direct binary downloads.
      - YOLOv8n is a better model (COCO 80 classes, higher accuracy) and its ONNX
        export is supported by the `ovc` tool bundled in the main openvino package.
      - Total size is comparable (~12 MB ONNX, ~6 MB IR after FP16 compression).

    Requires: ultralytics  (pip install ultralytics)
    """
    import subprocess, sys, shutil

    ir_dir = MODELS_DIR / "openvino" / "public" / "mobilenet-ssd" / "FP32"
    ir_xml = ir_dir / "mobilenet-ssd.xml"
    if ir_xml.exists():
        print("[openvino] IR model already exists, skipping download.")
        return

    ir_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = MODELS_DIR / "openvino" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "yolov8n.onnx"

    print("[openvino] Downloading YOLOv8n COCO weights via ultralytics (~6 MB)...")
    # ultralytics exports directly to ONNX; downloads .pt weights on first run.
    result = subprocess.run(
        [sys.executable, "-c",
         "from ultralytics import YOLO; import shutil, pathlib; "
         "m = YOLO('yolov8n.pt'); m.export(format='onnx', imgsz=640, opset=11); "
         f"shutil.move('yolov8n.onnx', r'{onnx_path}')"],
        capture_output=False,
    )
    if result.returncode != 0 or not onnx_path.exists():
        print("[openvino] ERROR: ONNX export failed. Install ultralytics: pip install ultralytics")
        return

    print("[openvino] Converting ONNX -> OpenVINO IR (FP16)...")
    result = subprocess.run(
        [sys.executable, "-m", "openvino.tools.ovc",
         str(onnx_path), "--output_model", str(ir_dir / "mobilenet-ssd")],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[openvino] ERROR: ovc conversion failed.")
        return

    print(f"[openvino] Done. IR saved to: {ir_dir}")


def download_easyocr():
    print("[easyocr] Downloading EasyOCR English model (~100 MB)...")
    import easyocr
    easyocr_dir = MODELS_DIR / "easyocr"
    easyocr_dir.mkdir(exist_ok=True)
    # Redirect model storage out of ~/.EasyOCR into our models/ dir
    easyocr.Reader(["en"], model_storage_directory=str(easyocr_dir),
                   download_enabled=True)
    print("[easyocr] Done.")


if __name__ == "__main__":
    print(f"Models will be stored in: {MODELS_DIR.resolve()}")
    print("Estimated total download: ~300 MB (whisper ~150 MB, YOLOv8n ~6 MB pt + ~12 MB ONNX, EasyOCR ~100 MB)\n")

    download_whisper()
    download_openvino_model()
    download_easyocr()

    print("\nAll models downloaded. The app is now fully offline.")
