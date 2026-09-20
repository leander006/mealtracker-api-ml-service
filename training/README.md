# Training pipeline

This folder lives inside the **ml-service** repo (`ml-service/training/`),
not as a separate repo — it writes trained weights directly into
`../weights/food_yolov8.pt`, which ml-service loads when `DETECTOR_MODE=real`.
Keeping it in the same repo means that path just works; if you ever want it
as its own repo instead, you'll need to adjust the `--out` default in
`train.py` and the `sys.path` line in `export_corrections.py` to point at
wherever you check out ml-service's `config.py`.

Two phases: (1) initial training from manually annotated photos, (2) ongoing
retraining from user corrections collected via the app's feedback loop.

## Phase 1 — initial model

This part cannot be automated for you — it requires real photos of your 15
food classes and someone to draw bounding boxes on them. Steps:

1. **Collect photos.** Aim for 150-300 images per class minimum, varied
   lighting/angle/plate/background. Fewer than ~100/class and the model
   will overfit hard.
2. **Annotate.** Use [Roboflow](https://roboflow.com) (free tier, browser-based,
   exports directly to YOLOv8 format - easiest path) or
   [LabelImg](https://github.com/HumanSignal/labelImg) (local, more manual).
   Draw one box per food item per photo, labeled with the matching class name
   from `data.yaml`.
3. **Export** into this structure under `training/dataset/`:
   ```
   dataset/
     images/train/*.jpg
     images/val/*.jpg
     labels/train/*.txt   # YOLO format, one .txt per image, same filename
     labels/val/*.txt
   ```
4. **Fill in `data.yaml`** — it currently has placeholders for 13 of your 15
   classes; add the 2 missing "original class" names before training (see
   the NOTE comment in that file).
5. **Train:**
   ```bash
   pip install -r requirements.txt
   python train.py --data data.yaml --epochs 100 --base yolov8n.pt
   ```
   This writes the best weights to `../weights/food_yolov8.pt`
   automatically. On CPU (e.g. a laptop or free-tier EC2), expect this to
   take hours, not minutes, for 100 epochs on a few hundred images/class —
   yolov8n is the right choice for CPU training, don't reach for a larger
   base model without a GPU.
6. Set `DETECTOR_MODE=real` in `ml-service/.env` and restart the ml-service.

Check the printed `mAP50` after training — that's the standard detection
accuracy metric (mean average precision at 50% IoU overlap). Above ~0.5 is a
reasonable starting point for a real-world demo; don't expect 0.9+ without a
lot more data than a first pass will have.

## Phase 2 — retraining from corrections

Once the app is live and users are scanning meals, every confirmed/corrected
prediction becomes a training row across your round-robin training DBs. To
fold that back into the model:

```bash
python export_corrections.py --out dataset/retrain_batch_1 --data-yaml data.yaml
python train.py --data data.yaml --resume-from ../weights/food_yolov8.pt --epochs 30
```

Read `export_corrections.py`'s module docstring before relying on this —
corrections currently carry a label and weight but not a hand-drawn bounding
box, so the exporter approximates one. It's good enough to reinforce label
accuracy; it will NOT meaningfully improve localization (box tightness). If
you want the model's box accuracy to actually improve from corrections
later, the next investment is a simple bounding-box draw step in the Scan
UI's correction flow, not more of this approximation.

Repeat this cycle periodically (weekly/monthly depending on volume) rather
than on every single correction — small-batch retrains on a handful of new
images risk overfitting to recent noise.
