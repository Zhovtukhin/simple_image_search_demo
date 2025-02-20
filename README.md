# Visual Product Search - Technical Test Task

To start test app run: 
> `streamlit run streamlit_app.py`

Demo in the end of README
Before running please download csv data and model from [here](https://drive.google.com/drive/folders/18-BBC2XoVeBidGmKEAWAK1coXuwPX7AF?usp=sharing)

## Project Structure:

    .
    ├── models/                             # Folder with models
    │   ├── cluster_classifier_crop.pkl     # SVC model for classification before search
    │   ├── cluster_classifier_padding.pkl  # Same but another image preprocessing
    │   ├── paddle_emb.onnx                 # Paddle embedder
    │   ├── vit_emb.onnx                    # Small ViT embedder
    ├── data/                               # Folder with cvs files
    │   ├── image_emb_crop_paddle.csv       # Dataset for paddle embedder
    │   ├── image_emb_padding_paddle.csv    # Same but another image preprocessing
    │   ├── image_emb_crop_ViT.csv          # Dataset for ViT embedder
    │   ├── image_emb_padding_ViT.csv       # Same but another 
    │   ├── text_emb_ViT.csv                # Embeddings for text classes
    │   ├── test_image.jpg                  # Test image 
    ├── streamlit_app.py                    # Main script
    ├── image_retriever.py                  # File with class of image recognizer
    ├── pathes.json                         # File with predifiended inputs
    ├── preparations.ipynb                  # Jupyter notebook with model convertions to onnx, database creation and models training
    ├── utils.py                            # Additional functions
    ├── requirements.txt
    └── README.md


## App Functionality:

There are several features that can be selected to see the result:
- Feature extractor method
- Preprocessing method
- Target image augmantation
- Classification before search
- How many similar image to show
- Use PCA or not
- What fetures left after PCA 
- Base and Target Image paths

First choose some fetures and insert paths of base and target image. Than use `Run Similarity Search` Button to sgor results of search. Each step also show their fps. In the end of the app some info about used resourses shows.

### Feature extractors
Available 2 feature extractors
- Small ViT transformer from [Tiny Clip](https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M). Input 224x224. Output: 512 features
- Embedder from [Paddle Image Recognition](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/en/quick_start/quick_start_recognition_en.md). Input 224x224. Output: 512 features

### Preprocessing methods
Available 2 methods
- Padding: Resize image in way that larger side 224 and than padding to rectangle in smaller side 
- Rect crop: crop center rectangle of image 

### Augmantations 
If use augmantations final serach will use all augmented images and return top K results. Classification will use mode. Apply this transforms:
- Gaussian Blur
- Sharpening
- Add noise
- Horizontal Flip
- Vertical Flip
- Increse brightness
- Decrease brightness

### Classification
First classify target image and than use only images with this class from base to search similar images. For each feature extractor using different classification method:
- ViT: Use text embeddings of 10 possible classes ["Person", "Group of People", "Clothing", "Electronic device", "Furniture", 'Animals', 'Landscape', 'City', 'Plants', 'Vehicle'] and calculate cosine similarity with target image embedding. Text embeddings created by same CLIP model as image
- Paddle: First base embedings calculated by paddle model was separated by 10 clusters using KMeans algo. Than predicted clusters used as labels to tran SVC model on this embeddings. This model use for predicing target image class.

### PCA
Before image search PCA can be applied to decrease dimentionality of embeddings space in order to speed up serach. 

### Base and Target Image paths
 Should be the full path to image and folder with unzip base images. Under the input field appeared info about valid path or not.

### Similarity Search
Realized using cosine similarity. Target image converted in embeding using some model and than it use to find most similar with base. Base created by the same model and preprocessing method. If using aumentation for each image top K images selected and than choose K base images with bigger score among all.

## Possible improvments:
- Use CUDA to run model faster (TensorRT for example)
- Realize one more type of search using keypoints and discriptors (SIFT for example) for database and matching. Probably will be slow due to matching target image with all base.
- Use FAISS library to increase search speed. Dont use it because it use Euclidian distance by defoult
- Apply PCA using FAISS
- Use bigger models to get better embeddings
- Use more augmantations
- Use CNN for classification before search


## Demo

[![DEMO]](https://raw.githubusercontent.com/Zhovtukhin/simple_image_search_demo/main/assets/demo.mp4)