### README: Tomato Leaf Disease Detection Using CNN  

---

#### **Project Overview**  
This project is a deep learning-based solution for classifying tomato leaf diseases using Convolutional Neural Networks (CNN). The model is trained to identify and classify various diseases affecting tomato leaves. The dataset used is sourced from Kaggle and contains images of tomato leaves with different diseases and healthy samples.  

---

#### **Dataset**  
The dataset used for this project is publicly available on Kaggle:  
[Tomato Leaf Disease Dataset](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf)  

The dataset includes images of tomato leaves categorized into several classes, including:  
- Bacterial spot  
- Early blight  
- Late blight  
- Leaf mold  
- Septoria leaf spot  
- Two-spotted spider mites  
- Target spot  
- Tomato yellow leaf curl virus  
- Tomato mosaic virus  
- Healthy  

---

#### **Key Features**  
- **Dataset Preprocessing**:  
  - Resized all images to a uniform size (e.g., 224x224).  
  - Normalized pixel values for better convergence during training.  
  - Augmented images to improve model generalization.  

- **Model Architecture**:  
  - A custom CNN architecture with multiple convolutional, pooling, and fully connected layers.  
  - Utilized ReLU activation functions and softmax for multi-class classification.  

- **Training**:  
  - Optimizer: Adam  
  - Loss Function: Categorical Crossentropy  
  - Metrics: Accuracy  

- **Evaluation**:  
  - Achieved high accuracy on the test dataset.  
  - Visualized performance using confusion matrices, classification reports, and accuracy/loss curves.  

---

#### **Installation**  
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```  
2. Install required Python packages:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf?resource=download) and place it in the `input/data/` folder.

---

#### **Usage**  
1. **Training the Model**:  
   Run the training script:  
   ```bash
   python train.py
   ```  

2. **Evaluating the Model**:  
   Run the evaluation script:  
   ```bash
   python evaluate.py
   ```  

3. **Predicting on New Images**:  
   Use the prediction script with a sample image:  
   ```bash
   python predict.py --image <path_to_image>
   ```  

---

#### **Results**  
- **Training Accuracy**: ~93.3%  
- **Testing Accuracy**: ~88.1%  

Detailed results, including confusion matrices and visualizations, are saved in the `input/` folder.

---

#### **References**  
- Dataset: [Tomato Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf?resource=download)  
- CNN concepts and implementation: Deep Learning resources from TensorFlow and Keras documentation.

---

#### **Future Work**  
- Expand the dataset to include more diverse leaf images.  
- Experiment with pre-trained models (e.g., ResNet, EfficientNet) for transfer learning.  
- Develop a mobile application for real-time disease detection.  

---  

#### **Author**  
Gokarakonda Nikhil Sri Sai Teja  
