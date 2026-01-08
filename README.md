<div align="center">

[![Paper](https://img.shields.io/badge/paper-IEEE%20ICONS--IoT%202025-blue.svg)](https://ieeexplore.ieee.org/document/11211242)

# Edge AI-Based Defect Detection in White Pepper (Piper Nigrum L.) Using CLAHE-Based Preprocessing and YOLO

Faturrahman Syauqi (Universitas Syiah Kuala), [Kahlil Muchtar (Universitas Syiah Kuala, COMVISLAB USK)](https://comvis.mystrikingly.com/), Safrizal Razali (Universitas Syiah Kuala), Maulisa Oktiana (Universitas Syiah Kuala), and Al Bahri (Universitas Syiah Kula)<br>
</div>

---

<div align="justify">

> **Abstract:** _The quality sorting of white pepper is essential in agriculture, especially to support Indonesia’s position as a major pepper exporter. Existing defect detection models face limitations under poor lighting conditions, affecting the accuracy of automated sorting. This study proposes a YOLOv8-based defect detection model integrated with the Edge AI device NVIDIA Jetson Orin Nano. To enhance image contrast and improve detection under varying lighting, CLAHE pre-processing is applied. The dataset includes images of white pepper classified into two categories: normal and defective. The proposed system performs real-time detection with low latency, leveraging the Jetson Orin Nano’s on-device processing capabilities for efficiency and mobility. A Streamlit-based interface is also developed for real-time visualization. Evaluation results demonstrate improvements across all metrics when using CLAHE, achieving accuracy, precision, recall, specificity, and F1-score of up to 99%, and an mAP50-95 of 82%. Additionally, the IoU scores exceeding 90% indicate a high level of detection accuracy. The results aim to support AI-driven solutions in agricultural quality control._

</div><br>

<p align="center">
  <img style="width: 70%" src="Media/Fig 1.jpg">
</p>

<small>_Fig. 1. The overall system pipeline, including dataset preprocessing, Image preprocessing, Model training, Model evaluation, Edge deployment, and Web-Based visualization._</small>
<br><br>

<p align="center">
  <img style="width: 70%" src="Media/Fig 2.jpg">
</p>

<small>_Fig. 2. Presents pepper images before and after the preprocessing stage._</small>

---

## 📊 Data

Please download the ISIC and USK-Normal Skin datasets, available as original images and CLAHE-enhanced images. Both datasets include predefined training, validation, and test splits.

- **Original Dataset:**  
  [🔗 Google Drive Link](https://drive.google.com/drive/folders/1k0nsna_d6lRUv_Ght7VFkYTs4LheC3dq?usp=sharing)

- **CLAHE-Enhanced Dataset:**  
  [🔗 Google Drive Link](https://drive.google.com/drive/folders/1v7lAMdUK5UISLCsfJGfoIl9e1Wnb_pUy?usp=sharing)

---

## ⚙️ Hyperparameters

<p align="center"><b>Table 1. Hyperparameter Settings For Model Training</b></p>
<div align="center">
  <small>
    <table >
        <tr style="background-color:#b3b3b3; text-align:center;">
            <th>Parameter</th>
            <th>Without CLAHE</th>
            <th>With CLAHE</th>
        </tr>
        <tr>
            <td>
            Epoch
            </td>
            <td>50</td>
            <td>50</td>
        </tr>
         <tr>
            <td>
            imgsz
            </td>
            <td>224</td>
            <td>224</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>16</td>
            <td>16</td>
        </tr>
        <tr>
            <td>Optimizer</td>
            <td>AdamW</td>
            <td>AdamW</td>
        </tr>
        <tr>
            <td >Learning Rate</td>
            <td>0.0001</td>
            <td>0.0001</td>
        </tr>
        </tr>
    </table>
  </small>
</div>

---

## 🚀 How to Run (Google Colab)

This project is designed to be trained using **Google Colab with GPU acceleration**.

### 1️⃣ Open Google Colab
- Create a new notebook
- Go to **Runtime → Change runtime type → GPU**

### 2️⃣ Install Dependencies
```python
!pip install ultralytics torch torchvision
```

### 3️⃣ Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4️⃣ Train YOLOv8 Model
```python
from ultralytics import YOLO
import torch

# Load YOLOv8 model
model = YOLO('yolov8m.pt')

results = model.train(
    data='/content/drive/MyDrive/TA/Dataset/Biji Lada Lengkap 224x224/data.yaml',
    epochs=50,
    imgsz=224,
    batch=16,
    optimizer='AdamW',
    lr0=0.0001
)

print("\nTraining completed!\n")

```
---
## 📈 Results

<p align="center"><b>Table 1. Evaluation Metric Results</b></p>
<div align="center">
  <small>
    <table >
        <tr style="background-color:#b3b3b3; text-align:center;">
            <th>Metric</th>
            <th>Wihout CLAHE</th>
            <th>With CLAHE</th>
        </tr>
        <tr style="text-align:center;" >
            <td>
            Accuracy
            </td>
            <td>98%</td>
            <td>99%</td>
        </tr>
        <tr style="text-align:center;">
            <td>Precision</td>
            <td>98%</td>
            <td>99%</td>
        </tr>
        <tr style="text-align:center;">
            <td >Recall</td>
            <td>98%</td>
            <td>99%</td>
        </tr>
        <tr style="text-align:center;">
            <td >Specificity</td>
            <td>97%</td>
            <td>99%</td>
        </tr>
        <tr style="text-align:center;">
            <td>F1-Score</td>
            <td>98%</td>
            <td>99%</td>
        </tr>
        <tr style="text-align:center;">
            <td>mAP50-95</td>
            <td>79%</td>
            <td>82%</td>
        </tr>
    </table>
  </small>
</div>
<br>

<p align="center">
  <img style="width: 60%" src="Media/Fig 3.png">
</p>

<small>_Fig. 3. Bar chart comparison of classification metrics(accuracy, precision, recall, specificity, F1-score, mAP50-95) for YOLO with and without CLAHE._</small>
<br><br>

<p align="center">
  <img style="width: 80%" src="Media/Fig 4.png">
</p>

<small>_Fig. 4. IoU Score Results for Normal and Defective Classes._</small>

---

## 🎨 Qualitative Results
<p align="center">
  <img style="width: 50%" src="Media/Fig 5.jpg">
</p>
<p align="center">
  <img style="width: 50%" src="Media/Fig 6.jpg">
</p>

<small>_Fig. 5. Streamlit interface deployed on NVIDIA Jetson Orin Nano. The interface allows users to upload images and view real-time detection results, including prediction probabilities and inference time._</small>

---

## 📝 Citation

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{faturrahman2025edgeai,
  title={Edge AI-Based Defect Detection in White Pepper (Piper Nigrum L.) Using CLAHE-Based Preprocessing and YOLO},
  author={Faturrahman Syauqi, Kahlil Muchtar, Safrizal Razali, Maulisa Oktiana, Al Bahri},
  booktitle={2025 IEEE International Conference on Networking, Intelligent Systems, and IoT (ICONS-IoT)},
  year={2025},
  doi={10.1109/ICONS-IoT65216.2025.11211242}
}
```