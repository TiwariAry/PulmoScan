# 🫁 PulmoScan – Pneumonia Detection & Grad-CAM Visualizer

**PulmoScan** is a deep learning-powered web app that detects **pneumonia from chest X-ray images** using a MobileNetV2 model, and visualizes the key decision areas using **Grad-CAM** heatmaps.

🌐 **Live App:** [PulmoScan](https://web-production-96f6.up.railway.app/)
<br/>

![Screenshot 2025-06-23 224845](https://github.com/user-attachments/assets/57d430be-fdaa-495d-be9b-62bedd8d447f)
![Screenshot 2025-06-23 225013](https://github.com/user-attachments/assets/c9ab8fef-8dde-4bc1-b013-186c362b0ea6)

---

## 🔍 Features

- 📷 Upload chest X-ray images
- 🧠 Predicts **Pneumonia** or **Normal** using a trained CNN
- 🔥 Visualizes attention regions using **Grad-CAM**
- 🌐 Simple web interface with confidence score and visual overlays

---

## 🛠️ Tech Stack

- **Frontend**: HTML + Jinja templates
- **Backend**: Python Flask
- **ML Framework**: TensorFlow 2.19 (Version is important)
- **Model**: Pretrained MobileNetV2 fine-tuned on chest X-rays
- **Visualization**: Grad-CAM with OpenCV
- **Deployment**: RailwayCLI

---

## 🧠 Model Training Notebook

Model was trained on crop recommendation dataset.
Please visit the colab notebook to see the training process.

📓 [Google Colab Notebook](https://colab.research.google.com/drive/1WOU61YI0zRd58TNSZO39Tnt7YC_dcsj5?usp=sharing)

Below is the link to the dataset on which the model was trained.

📓 [Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🌿 System Overview

![image](https://github.com/user-attachments/assets/0b65ad62-342d-4b4e-88ad-7387ca1a4994)

---

## 🧠 Model Details

- Base Model: `MobileNetV2`
- Input Size: `224x224x3`
- Layers fine-tuned: Top classifier layers
- Trained on: Pneumonia vs Normal images (Chest X-ray dataset)
- Accuracy: ~**88.14%** on test set

---

## 📂 Project Structure

```bash
PulmoScan/
│
├── static/uploads/ # Uploaded images + visualizations
├── templates/
│ ├── index.html # Upload page
│ └── result.html # Results page
├── model_compatible.h5 # Trained MobileNetV2 model
├── app.py # Main Flask app
├── requirements.txt
└── README.md
```

---

## 🚀 How It Works

### 🩻 Input & Prediction
- User uploads a chest X-ray image through the web interface.
- The image is resized, normalized, and passed to the trained MobileNetV2 model.
- The model outputs a binary classification: Pneumonia or Normal with confidence score.

### 🔥 Grad-CAM Visualization
- Using TensorFlow’s GradientTape, a heatmap is generated from the last convolutional layer.
- Gradients are backpropagated to highlight regions that influenced the model’s decision.
- The heatmap is blended with the original image using OpenCV.

### 📄 Display Results
The results page displays:
- Original X-ray
- Grad-CAM heatmap
- Overlay image
- Predicted label and confidence
Users can visually interpret the model's focus regions in its diagnosis.

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/PulmoScan.git
cd PulmoScan
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the App
```bash
python app.py
```

Visit *http://localhost:5000* in your browser.

---

## 📦 Deployment Notes
- For Render: TensorFlow version is restricted (max 2.12)
- For Railway: Full support for TensorFlow 2.19

---

## 🧠 Learnings & Highlights
- Fine-tuned and deployed MobileNetV2 on pneumonia detection dataset
- Applied Grad-CAM to interpret CNN decisions on medical images
- Built a responsive, user-friendly web interface using Flask, HTML/CSS, and Jinja2
- Handled file uploads, image preprocessing, and prediction pipelines in production
- Deployed the full-stack application on Railway, managing TensorFlow version constraints

---

## 📣 Future Enhancements
- 🧪 Add support for multi-class classification (e.g., COVID-19, viral vs bacterial pneumonia)
- 🧠 Improve explainability with Layer-wise Relevance Propagation (LRP) or Integrated Gradients
- 🗂️ Include a patient history dashboard to store previous predictions
- 🔐 Add user authentication for medical professionals
- 🌐 Add API endpoint support for integrating with external diagnostic systems

---

## 🤝 Contributing
Pull requests are welcome! For significant changes, please open an issue first to discuss what you would like to improve or propose.
Let’s make PulmoScan more accessible, transparent, and useful for healthcare. 🫁

---

## 📄 License

This project is licensed under the [MIT License](https://github.com/TiwariAry/PulmoScan/blob/main/LICENSE).  
Feel free to fork, modify, and build on it.

---

## 👨‍💻 Author

**Aryan Tiwari**  
📫 [LinkedIn](https://www.linkedin.com/in/aryan-tiwari-6844a9250)  
💻 [GitHub](https://github.com/TiwariAry)

---
