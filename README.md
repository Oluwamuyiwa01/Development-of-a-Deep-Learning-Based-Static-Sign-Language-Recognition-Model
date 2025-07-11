# Development-of-a-Deep-Learning-Based-Static-Sign-Language-Recognition-Model
This project presents a Convolutional Neural Network (CNN) developed for recognizing static sign language gestures. It was carried out as my undergraduate thesis in Computer Science at Federal University Lokoja, Nigeria.

## Overview.

This project presents a deep learning-based static sign language recognition model designed to assist the deaf and hard-of-hearing in basic communication. The system captures and classifies static hand gestures using a convolutional neural network (CNN) trained on a self-generated dataset. Key components include a data acquisition module using a webcam, preprocessing techniques for noise reduction and segmentation, feature extraction using CNNs, and model testing for performance tuning. The model can serve as both an educational tool and a foundational communication aid.  
## Data Collection and Preprocessing.  
* Type: Image dataset of static hand signs.  
* Number of classes: 12.  
* Source: Self-generated using OpenCV via webcam.  
* Collection Method: Captured multiple images per class using OpenCV in real-time with hand tracking and segmentation.
* Preprocessing: Resizing, grayscale conversion, normalization, and one-hot encoding.  
* Dataset split: 80% training, 20% testing.  
<img width="639" height="259" alt="Vi" src="https://github.com/user-attachments/assets/86168693-21d3-42bf-8a70-2b7251904229" />

## Dataset.  

To access the dataset I used in this project, please request permission via this link: [Download Dataset](https://drive.google.com/file/d/https://drive.google.com/file/d/1QdowjuEHzlpvslyBXKdpV0t51wWD0lee/view?usp=drive_link/view?usp=sharing)  

## Model Architecture.    
* I built the model using TensorFlow/Keras      
* Input image size: 224 X 224   
* Layers: Conv2D, MaxPooling2D, Dropout, Flatten, Dense    
* Loss: Categorical crossentropy; Optimizer: Adam   
* Trained for 20 epochs    
* Achieved high accuracy and generalization  
  
## Performance Evaluation.  
<img width="441" height="146" alt="accuracy" src="https://github.com/user-attachments/assets/12d3930b-4186-4f2c-ad0f-b5e1ef7a692e" />  

## Confusion matrix.  
<img width="550" height="437" alt="matr" src="https://github.com/user-attachments/assets/e855d9e5-1e44-4116-b207-63c7375b6c70" />  

## Presentations of some of the results.    
<img width="644" height="484" alt="C" src="https://github.com/user-attachments/assets/285a7d6d-d555-4f2d-9d0e-4ad2c7a5e24d" />  
<img width="638" height="481" alt="Y" src="https://github.com/user-attachments/assets/70c1495b-605f-40e4-840b-536de7f46c7e" />  
<img width="1366" height="768" alt="Screenshot (32)" src="https://github.com/user-attachments/assets/4f71ac7a-78db-46d6-b11b-0cad899b013c" />  

## Contribution.  
This project presents a deep learning-based static sign language recognition model that enhances basic communication for individuals with hearing impairments. Using Convolutional Neural Networks (CNNs) and advanced preprocessing techniques, the system accurately classifies static hand gestures in real-time. It is optimized for speed and efficiency, making it suitable for deployment on standard hardware. Beyond its practical impact, the project addresses gaps in existing models and provides a solid foundation for future work on dynamic gestures and multi-language sign support. It also contributes academically by offering a detailed case study for researchers in computer vision and assistive technology.   

## Download Trained Model.

The trained model file (`CNN-MODEL.h5`) is available for **live implementation on standard hardware**, as well as for **academic** and **research** purposes. link: [Trained Model](https://drive.google.com/drive/folders/13ekzOaJyp7E-Y8Yf3DUWKcBTG5z05QMd?usp=drive_link)  

## How to Run the Flask App.
Follow the steps below to run the sign language recognition model using the Flask web framework:
- Clone the Repository.
- Install dependencies. Make sure you have Python installed (Python 3.7+ is recommended), then install all required packages.
- Download the trained .h5 model from Google Drive: [link](https://drive.google.com/drive/folders/13ekzOaJyp7E-Y8Yf3DUWKcBTG5z05QMd?usp=drive_link) 
- After download, place the CNN-MODEL.h5 file in the root directory of the project (same folder as app.py).
- Run the Flask app using `python app.py` in your terminal.
- Open a browser, and type this address http://127.0.0.1:5000/ to interact with the sign language recognition interface.
