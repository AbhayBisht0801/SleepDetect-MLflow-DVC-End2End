# Sleep Detection Project Readme

## Overview

This project aims to detect sleep stages using stable diffusion for image generation, a VGG16-based model for classification, MLflow for model tracking, DVC for pipeline tracking, and Streamlit for deployment in a local host environment.


### Data Preparation

1. Ensure that your dataset is properly formatted and organized. This may involve preprocessing images or videos to extract relevant frames.

2. Organize your data into appropriate directories, such as `train`, `validation`, and `test`.

### Model Training

1. Run the data preprocessing scripts if necessary to generate input data for training.

2. Train the VGG16 model using the provided scripts or your custom implementation.
3. Monitor training progress using MLflow.

### Model Evaluation

1. Evaluate the trained model on the test dataset.

### Deployment

1. Deploy the model using Streamlit for local hosting.

2. Access the deployed application via your web browser at `http://localhost:8501`.

## Acknowledgements

- This project utilizes the VGG16 model architecture.
- MLflow is used for model tracking.
- DVC is used for pipeline tracking.
- Streamlit is used for local deployment.

