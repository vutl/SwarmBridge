"""R1a onboard detector (reports/archive/r1-visdrone/swarmbridge-r1a.md 0.2): YOLO11n, COCO-pretrained, trained on the SAME prepared
VisDrone-DET-train split as the A5 detector (a5_det.py prep: train sequences shared with DET-val dropped, 10 % of
sequences held out), imgsz 640, 40 epochs, batch 16, seed 0, deterministic, default augmentation; best.pt by
holdout fitness. Usage: python r1a_det.py PREP_DIR RUN_DIR
"""
import sys
from pathlib import Path


def main():
    from ultralytics import YOLO
    prep, run = sys.argv[1], sys.argv[2]
    m = YOLO("yolo11n.pt")
    m.train(data=str(Path(prep) / "data.yaml"), imgsz=640, epochs=40, batch=16, workers=3, seed=0, deterministic=True,
            project=run, name="yolo11n_visdrone_640", exist_ok=True, plots=False, cache=False, amp=True)


if __name__ == "__main__":
    main()
