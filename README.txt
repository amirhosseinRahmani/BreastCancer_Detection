BREAST CANCER DETECTOR - PyQt6
===============================

This version uses PyQt6 instead of Streamlit.

The CNN architecture is kept compatible with the BUSI CNN project:
- Input: grayscale 400x400
- Conv: 1 -> 32
- Conv: 32 -> 64
- Conv: 64 -> 128
- MaxPool after each convolution
- Linear: 128*50*50 -> 256
- Dropout: 0.5
- Output: 3 classes

Classes:
1. benign
2. malignant
3. normal

SETUP
-----

1. Put your trained model file in this folder:

       busi_cnn.pth

2. Install dependencies:

       pip install -r requirements.txt

3. Run:

       python app.py

IMPORTANT
---------

The trained .pth model is not included because the model file was not
available in the saved project files. Put your own trained BUSI model
(state_dict) next to app.py.

The application itself has been syntax-checked and the CNN forward pass
has been tested with a 400x400 dummy tensor.
