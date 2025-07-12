
# Player Tracking and Re-Identification in Sports Video

A robust pipeline for tracking players in sports footage using YOLO for detection, DeepSORT for multi-object tracking, and a deep ReID model for appearance-based re-identification. This project is optimized for challenging, low-quality video and ensures consistent player IDs—even when players leave and re-enter the frame.

## 🚀 Features

- **Player Detection:** YOLOv8 model detects players in each frame.
- **Tracking:** DeepSORT assigns persistent IDs, maintaining player identity across frames.
- **Deep Re-Identification:** Integrates OSNet (Torchreid) for robust appearance-based matching, reducing ID switches and improving re-identification after occlusion or re-entry.
- **Fallback Appearance Features:** Uses color histograms if deep embeddings are unavailable.
- **Ghost Track Reduction:** Filters out short-lived and ghost tracks with `max_age` and minimum track age.
- **Visualization:** Draws bounding boxes, IDs, and (optionally) player trajectories.
- **Optimized for Low-Quality Video:** Tuned parameters and robust features handle missed detections and occlusions.


## 🛠️ Requirements

- Python 3.8+
- NVIDIA GPU (recommended for real-time performance)


### Install All Dependencies

All required packages are listed in `requirements.txt`:

```
ultralytics
opencv-python
numpy
deep-sort-realtime
torch
torchvision
torchreid
gdown
```

Install everything with:

```bash
pip install -r requirements.txt
```

> **Tip:** For GPU acceleration, install the CUDA-enabled version of PyTorch matching your GPU and driver. See [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## 📂 Usage

1. **Download the YOLO Model Weights**

Download the YOLO model weights (`best.pt`) from the following link:

[Download best.pt](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

Place the file in your project directory before running the script.
2. **Download the Input Video**

Download the 15-second input video (`15sec_input_720p.mp4`) from the following link:

[Download 15sec_input_720p.mp4](https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=drive_link)

Place the file in your project directory before running the script.
3. **Install all dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the tracking script:**

```bash
python track_players.py
```

5. **Press `q` to exit the video display window.**

## ⚙️ Key Parameters

| Parameter | Purpose | Typical Value |
| :-- | :-- | :-- |
| `max_age` | Frames to keep lost tracks before deletion | 7–15 (lower for noisy video) |
| `conf` threshold | Minimum detection confidence for tracking | 0.5–0.7 |
| `track.age` | Minimum frames before displaying a track | 3 |

## 🧩 Tips \& Troubleshooting

- **CUDA Errors:** If you don't have a GPU or CUDA-enabled PyTorch, remove all `.cuda()` or `.to(device)` calls and run on CPU.
- **Import Errors:** If you see `ModuleNotFoundError`, install the missing package with `pip install <package>`.
- **Performance:** For real-time speed, use a GPU and consider a lighter ReID model.


## 🌟 Further Enhancements

- **Save output video** with OpenCV’s `VideoWriter`.
- **Visualize player trajectories** for movement analysis.
- **Export tracking data** to CSV for analytics.
- **Fine-tune the ReID model** on your own sports data for even better re-identification.
- **Extend tracking** to other classes (goalkeeper, referee, ball) by adjusting class filters.


## 📄 License

This project is for educational and research purposes. Please respect the licenses of the underlying models and libraries you use.

## 🤝 Credits

- YOLO model: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Tracking: [DeepSORT](https://github.com/nwojke/deep_sort)
- ReID: [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)
- Video processing: [OpenCV](https://opencv.org/)


## 📬 Contact

For questions or improvements, open an issue or contact the project maintainer.

> **Ready to use!**
> Fork, star, and contribute to make this tracker even better!

<div style="text-align: center">⁂</div>

[^1]: image.jpg

[^2]: https://drive.google.com/file/d/1TDcND31fvEDvcnZCaianTxJrmT8q7iIi/view?usp=drive_link

