<div align="center">

# 👋 &nbsp;Jesús Royeth

### Computer vision engineer

`detection` &nbsp;·&nbsp; `segmentation` &nbsp;·&nbsp; `tracking` &nbsp;·&nbsp; `optimization` &nbsp;·&nbsp; `edge AI` &nbsp;·&nbsp; `cloud AI`

I work where computer vision models meet production: architectures, custom layers and heads,<br>
tracking pipelines, quantization, and deployment on constrained hardware.

<br>

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-8A2BE2?style=for-the-badge&logo=google&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

</div>

---

## 🔭 &nbsp;Open source

I contribute performance improvements, bug fixes, and hardware support to widely used computer
vision libraries. Recent work includes extending Hailo support across seven YOLO tasks, fixing
RKNN INT8 exports that produced zero detections, making an RF-DETR DataLoader up to 1.67× faster,
and reducing Hungarian matcher time by up to 77%.

| | Merged PRs |
|---|---|
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [106](https://github.com/ultralytics/ultralytics/pulls?q=is%3Apr+author%3AJESUSROYETH+is%3Amerged) |
| [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | [63](https://github.com/roboflow/rf-detr/pulls?q=is%3Apr+author%3AJESUSROYETH+is%3Amerged) |
| [roboflow/trackers](https://github.com/roboflow/trackers) | [15](https://github.com/roboflow/trackers/pulls?q=is%3Apr+author%3AJESUSROYETH+is%3Amerged) |
| [SpikeInterface/spikeinterface](https://github.com/SpikeInterface/spikeinterface) | [4](https://github.com/SpikeInterface/spikeinterface/pulls?q=is%3Apr+author%3AJESUSROYETH+is%3Amerged) |
| [roboflow/supervision](https://github.com/roboflow/supervision) | [2](https://github.com/roboflow/supervision/pulls?q=is%3Apr+author%3AJESUSROYETH+is%3Amerged) |

🏷️ &nbsp;My contributions have been credited by name in **39 upstream releases**.

---

## 🚀 &nbsp;Selected work

<details>
<summary><b>🟣 &nbsp;Ultralytics YOLO &nbsp;—&nbsp; 7 highlights</b></summary>

<br>

**Extended Ultralytics' Hailo backend from detection-only to seven tasks across the complete
`.pt → ONNX → INT8 → HEF` export path.**
[#25254](https://github.com/ultralytics/ultralytics/pull/25254) ·
[#25259](https://github.com/ultralytics/ultralytics/pull/25259) ·
[#25276](https://github.com/ultralytics/ultralytics/pull/25276) ·
[#25280](https://github.com/ultralytics/ultralytics/pull/25280) ·
[#25283](https://github.com/ultralytics/ultralytics/pull/25283) ·
[#25348](https://github.com/ultralytics/ultralytics/pull/25348)

**Fixed RKNN INT8 exports across detection, segmentation, pose, and OBB that produced valid files
but no detections.**
[#25524](https://github.com/ultralytics/ultralytics/pull/25524)

**Made channels-last the automatic default for native PyTorch inference on x86 CPUs, improving
median throughput by 18.43% for YOLO26n and 14.09% for YOLO11n.**
[#25983](https://github.com/ultralytics/ultralytics/pull/25983)

**Fixed dataset `fraction` sampling that selected images from a single class, raising YOLO11n-cls
top-1 accuracy from 9.89% to 87.64% in the validation run.**
[#25968](https://github.com/ultralytics/ultralytics/pull/25968)

**Restored ByteTrack's low-confidence recovery step, which was unreachable under the default
configuration and discarded detections in the 0.1–0.25 confidence range.**
[#25034](https://github.com/ultralytics/ultralytics/pull/25034)

**Removed per-object GPU stalls from result processing, making `Results.plot()` 7.55× faster and
cutting segmentation-validator synchronizations from 876 to 2.**
[#25230](https://github.com/ultralytics/ultralytics/pull/25230) ·
[#25853](https://github.com/ultralytics/ultralytics/pull/25853)

**Found train/test contamination affecting 27.9% of KITTI depth training images; validation found
no statistically significant accuracy impact.**
[#25650](https://github.com/ultralytics/ultralytics/pull/25650)

</details>

<details>
<summary><b>🟠 &nbsp;RF-DETR &nbsp;—&nbsp; 6 highlights</b></summary>

<br>

**Reduced the DataLoader handoff from 114 objects to 9 per batch, improving throughput by up to
1.67× and preventing file-descriptor crashes at high worker counts.**
[#1399](https://github.com/roboflow/rf-detr/pull/1399)

**Rewrote the evaluation confidence sweep from O(T·N) to O(N log N), making it 7.1–7.8× faster at
COCO scale with identical `macro_f1` results.**
[#1339](https://github.com/roboflow/rf-detr/pull/1339)

**Reduced Hungarian matcher time and peak CUDA memory by up to 77% by padding to the largest target
count in each batch instead of the total number of targets.**
[#1297](https://github.com/roboflow/rf-detr/pull/1297)

**Corrected the encoder mask-loss computation across all released segmentation models and added
the missing tests for `SegmentationHead`.**
[#1331](https://github.com/roboflow/rf-detr/pull/1331)

**Aligned ONNX/TFLite decoding with PyTorch, restoring 30 detections dropped across 26 of 200 COCO
validation images.**
[#1320](https://github.com/roboflow/rf-detr/pull/1320)

**Preserved callback state after resuming training, preventing diverged runs from overwriting good
checkpoints with NaN weights.**
[#1318](https://github.com/roboflow/rf-detr/pull/1318)

</details>

<details>
<summary><b>🔵 &nbsp;Trackers &nbsp;—&nbsp; 4 highlights</b></summary>

<br>

**Fixed a float-equality bug that degraded steady-frame-rate tracking by as much as 4.86 HOTA and
236 extra ID switches on DanceTrack.**
[#531](https://github.com/roboflow/trackers/pull/531)

**Fixed the CLI, demo, and benchmark paths silently assuming 30 fps—a default that was wrong for
every SportsMOT and DanceTrack sequence supported by the benchmark.**
[#569](https://github.com/roboflow/trackers/pull/569)

**Fixed OC-SORT omitting low-confidence inputs from its return value, restoring one output row per
detection.**
[#566](https://github.com/roboflow/trackers/pull/566)

**Reduced a Kalman filter noise-calculation hot path by 64.93% while preserving bit-identical
output.**
[#572](https://github.com/roboflow/trackers/pull/572)

</details>

---

## 🧰 &nbsp;Stack

**Frameworks & runtimes**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![JAX / Flax](https://img.shields.io/badge/JAX%20%2F%20Flax-8A2BE2?style=flat-square&logo=google&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TFLite-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)
![OpenVINO](https://img.shields.io/badge/OpenVINO-0071C5?style=flat-square&logo=intel&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

**Models & methods** &nbsp;&nbsp; YOLO (v8 / 11 / 26) · RF-DETR · DINOv2 · SAM · deformable DETR ·
Kalman filtering and multi-object tracking

**Silicon & platforms** &nbsp;&nbsp; Hailo · MemryX MXA · RKNN · Cloud TPU · Google Cloud

---

## 🎓 &nbsp;Certifications

![GCP ML Engineer](https://img.shields.io/badge/GCP-Professional%20ML%20Engineer-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
![GCP Cloud Architect](https://img.shields.io/badge/GCP-Professional%20Cloud%20Architect-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
![GCP Data Engineer](https://img.shields.io/badge/GCP-Professional%20Data%20Engineer-4285F4?style=flat-square&logo=googlecloud&logoColor=white)
