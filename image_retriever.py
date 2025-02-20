#
from sklearn.decomposition import PCA
import os
import cv2
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
from utils import letterbox
from sklearn.metrics.pairwise import cosine_similarity

# STD and MEAN for normalizing input data
MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(-1,1,1) 
STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(-1,1,1)

vit_path = "vit_emb.onnx"
paddle_path = "paddle_emb.onnx"

class Retriever:
    """
    Class for classification.
    Embedding model is used to generate embeddings for images. And compare it with base.
    """
    def __init__(self, model_type, preprocess_type, image_width=224, image_height=224,
                       models_folder='models', data_folder='data'):
        """

        :param model_type: str, with model to use<br>
        :param preprocess_type: str, how to preprocces data <br>
        :param image_width: int, image width for embeder model <br>
        :param image_height: int, height width for embeder model <br>
        """
        self.base = None
        self.preprocess_type = preprocess_type
        self.model_type = model_type
        self.model_path = ''
        if model_type == "ViT":
            self.model_path = os.path.join(models_folder, vit_path)
        elif model_type == "paddle":
            self.model_path = os.path.join(models_folder, paddle_path)
        else: # for keypoints matching
            self.model = cv2.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
            
        self.image_width = image_width  
        self.image_height = image_height 
        self.model = None
        self.apply_pca = False
        self.pca = None
        self.pca_n = None
        self.pca_base = None
       
        try:
            # read base
            base_file_name = os.path.join(data_folder, 
                                     f'image_emb_{preprocess_type.split()[-1]}_{model_type}.csv')
            self.base = pd.read_csv(base_file_name)
            
            if self.model_path:
                # load model
                self.model = ort.InferenceSession(self.model_path,
                                                  providers=['CPUExecutionProvider'])
                """embedding model"""
                self.model.disable_fallback()
                self.model_input_name = self.model.get_inputs()[0].name
                """embedding model input names"""
                self.model_output_name = self.model.get_outputs()[0].name
                """embedding model output names"""
                # warmup model
                self.apply_emb_model(np.ones((image_height, image_width, 3),
                                           dtype=np.uint8))
        except Exception as exc:
            print(f'Error image retriever init: {exc}')
            


    def preprocess(self, image):
        """
        Preprocess image for model<br>
        """
        if self.preprocess_type == 'rect crop':
            h, w = image.shape[:2]
            ratio = max(self.image_height / h, self.image_width / w)
            img_in = cv2.resize(image, (None, None),fx=ratio, fy=ratio, 
                                        interpolation=cv2.INTER_CUBIC)  # resize
            h, w = img_in.shape[:2]
            img_in = img_in[h//2 - self.image_height//2 : h//2 + self.image_height//2, 
                            w//2 - self.image_width//2 : w//2 + self.image_width//2]
        else:
            img_in = letterbox(image, (self.image_height, self.image_width), (0, 0, 0))[0]
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)  # convert to RGB

        img_in = img_in.astype(np.float32) / 255.0
        img_in = np.transpose(img_in, (2, 0, 1))
        img_in = (img_in - MEAN) / STD
        # img_in = img_in.astype(np.float16)
        return img_in

    
    def apply_kp_model(self, image, preprocess=True):
        """
        Apply sift to image or images<br>
        """
        model_input = []
        if (not isinstance(image, list) or 
           (isinstance(image, np.ndarray) and len(image.shape) == 3)):
            image = [image]
        output = []
        for img in image:
            if preprocess:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.model.detectAndCompute(img, None)
            output.append([keypoints, descriptors])
            
        return output
        
        
    def apply_emb_model(self, image, preprocess=True):
        """
        Apply model to image or images<br>
        """
        model_input = []
        if (not isinstance(image, list) or 
           (isinstance(image, np.ndarray) and len(image.shape) == 3)):
            image = [image]
        for img in image:
            if preprocess:
                img = self.preprocess(img)
            img_in = np.ascontiguousarray(img)
            model_input.append(img_in)

        onnx_input_image = {self.model_input_name: model_input}
        output, = self.model.run(None, onnx_input_image)
        return output
        
        
    def search_image_in_base(self, target_embs, metaclass, top_k):
        """
        Search images similar to target image embedding<br>
        """
        self.preprocess_type
        
        all_sims = []
        all_ids = []
        
        if metaclass is not None:
            base_masked = self.base[self.base.category == metaclass]
            base_masked = base_masked.reset_index(drop=True)
        else:
            base_masked = self.base
            
        if not self.apply_pca or self.pca is None or self.pca_base is None:
            sims = cosine_similarity(target_embs, base_masked.iloc[:, 1:-1].to_numpy())
        else:
            if metaclass is None:
                sims = cosine_similarity(self.pca.transform(target_embs), self.pca_base)
            else:
                sims = cosine_similarity(self.pca.transform(target_embs), 
                                         self.pca.transform(base_masked.iloc[:, 1:-1].to_numpy()))
            
        sort_ids = np.argsort(sims, axis=1)[:, -top_k:]
        for i, l in enumerate(sort_ids):
            all_sims.extend(sims[i, l])
            
        all_ids = sort_ids.flatten()
            
        return all_ids, all_sims
     
    def make_pca(self, left_fetures_num):
        """
        Apply PCA to dataset<br>
        """
        start_time = time.time()
        if len(self.base.columns) - 2 > left_fetures_num and \
           self.pca_n != left_fetures_num:
            self.pca = PCA(n_components=left_fetures_num)
            self.pca.fit(self.base.iloc[:, 1:-1].to_numpy())
            self.pca_base = self.pca.transform(self.base.iloc[:, 1:-1].to_numpy())
            self.pca_n = left_fetures_num
            self.apply_pca = True
            
        return time.time() - start_time

