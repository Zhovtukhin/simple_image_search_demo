import os
import cv2
import time
import json
import pickle
import psutil
import numpy as np
import pandas as pd
import streamlit as st
import albumentations as A
from utils import letterbox
from image_retriever import Retriever
from sklearn.metrics.pairwise import cosine_similarity


models_path = 'models' # folder with models
data_path = 'data' # folder with csv data
pathes_cache_path = 'pathes.json'  # file with predifined text inputs
image_path = os.path.join(data_path, 'test_image.jpg') # defolt image to use
base_path = 'test_task_data' # path to folder with base images

if os.path.exists(pathes_cache_path):
    with open(pathes_cache_path) as f:
        try:
            pathes_cache = json.load(f)
        except:
            pathes_cache = {}
        
else:
    pathes_cache = dict()

if 'image_path' not in pathes_cache:
    pathes_cache['image_path'] = image_path
    
if 'base_path' not in pathes_cache:
    pathes_cache['base_path'] = base_path
    

with open(pathes_cache_path, 'w') as f:
    json.dump(pathes_cache, f)

# Augmantations list 
aug_list = [A.GaussianBlur(blur_limit=(5, 5), p=1.0), 
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 0.5), p=1.0), 
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.5, 0.5), p=1.0),  # Blur, sharpen, noise
            A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]  # Rotate


@st.cache_resource()
def initialize_image_retriever(onnx_model_path, process_type):
    """
    Initialize class for image search<br>
    """
    print('initialize image_retriever')
    return Retriever(onnx_model_path, process_type)

@st.cache_data()
def load_image(img_path):
    """
    Load image using path<br>
    """
    print('load_image')
    im = None
    im_shape = ()
    try:
        im = cv2.imread(img_path)
        im_shape = im.shape[:2]
        if min(im_shape) > 500:
            im = cv2.resize(im, None, fx=500/min(im_shape), fy=500/min(im_shape))
            im_shape = im.shape[:2]
    except:
        return None, im_shape

    return im, im_shape

@st.cache_data
def apply_augmantation(img):
    """
    Augment image<br>
    """
    print('apply_augmantation')
    progress_text = "Augmantation Processed. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    start_time = time.time()
    aug_len = len(aug_list) + 3
    processed_aug = 0
    
    images_aug = [img]
    for aug in aug_list:
        images_aug.append(aug(image=img)["image"])
        processed_aug += 1
        progress_bar.progress(processed_aug / aug_len, text=progress_text)
    
    images_aug.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    processed_aug += 1
    progress_bar.progress(processed_aug / aug_len, text=progress_text)
    
    for factor in [50, -50]: # change brightness
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, factor)
        v[v > 255] = 255
        v[v > 255] = 0
        final_hsv = cv2.merge((h, s, v))
        images_aug.append(cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR))
        processed_aug += 1
        progress_bar.progress(processed_aug / aug_len, text=progress_text)
   
    fps = time.time() - start_time
    fps = round(aug_len/fps, 3) 
    
    return images_aug, fps

@st.cache_data
def visualize_augmantation(images, fix_height=100, max_width=1000):
    """
    Create one image to show with all augmantations<br>
    """
    print('visualize_augmantation')
    rect_size = 3
    rows_cols_image = [[]]
    max_row_width = 0

    for i, img in enumerate(images):
        new_img = img
        if img.shape[0] > fix_height:
            new_x = int(img.shape[1] * (fix_height / img.shape[0]))
            new_img = letterbox(img, (fix_height, new_x), (0, 0, 0))[0]

        if (max_row_width + new_img.shape[1]) > max_width:
            rows_cols_image.append([new_img])
            max_row_width = new_img.shape[1]
        else:
            rows_cols_image[-1].append(new_img)
            max_row_width += new_img.shape[1]

    image = np.zeros((fix_height * len(rows_cols_image),
                      max([sum([col.shape[1] for col in row]) for row in rows_cols_image]), 3), dtype=np.uint8)
    last_x, last_y = 0, 0
    for row in rows_cols_image:
        max_y = max([col.shape[0] for col in row])
        for col in row:
            if col.shape[0] < max_y:
                col = letterbox(col, (max_y, col.shape[1]), (0, 0, 0))[0]
            image[last_y: last_y + max_y, last_x: last_x + col.shape[1]] = col
            last_x += col.shape[1]
        last_y += max_y
        last_x = 0
    image = image[:last_y]

    if image.shape[1] > max_width:
        image = cv2.resize(image, (max_width, int(image.shape[0] * (max_width / image.shape[1]))))
    return image


@st.cache_data
def apply_preprocess(images, prepr_type):
    """
    Preprocess image<br>
    """
    print('preprocess_images')
    if image_retriever is None:
        return [], 'NA'
        
    progress_text = "Image Preproccessed. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    start_time = time.time()
    imgs_len = len(images)
    processed_im = 0
    prepr_images = []
    
    for im in images:
        print(im.shape)
        prepr_images.append(image_retriever.preprocess(im))
        print(prepr_images[-1].shape)
        processed_im += 1
        progress_bar.progress(processed_im / imgs_len, text=progress_text)
        
    fps = time.time() - start_time
    fps = round(imgs_len/fps, 3)
    
    return prepr_images, fps
    
@st.cache_data
def apply_model(images, model_type):
    """
    Apply feture extractor to get embeddings<br>
    """
    print('apply_model')
    if image_retriever is None:
        return [], 'NA'
        
    progress_text = "Model Running. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    start_time = time.time()
    imgs_len = len(images)
    processed_im = 0
    embeds = []
    
    for im in images:
        if model_type != 'keypoints':
            embeds.append(image_retriever.apply_emb_model(im, preprocess=False)[0])
        else:
            embeds.append(image_retriever.apply_kp_model(im, preprocess=True)[0])
        processed_im += 1
        progress_bar.progress(processed_im / imgs_len, text=progress_text)
    embeds = np.asarray(embeds)
        
    fps = time.time() - start_time
    fps = round(imgs_len/fps, 3)
    
    return embeds, fps

@st.cache_data
def find_metaclass(embs, feature_extr_method, prepr_type):
    """
    Find metaclass of target embeddings<br>
    """
    print('find_metaclass')
    if image_retriever is None:
        return []
    
    start_time = time.time()
    imgs_len = len(embs)
    
    classes = []
    
    if feature_extr_method == 'ViT':
        text_embs = pd.read_csv(os.path.join(data_path, 'text_emb_ViT.csv'))
    elif feature_extr_method == 'paddle':
        with open(os.path.join(models_path, 
                               f"cluster_classifier_{prepr_type}.pkl", "rb")) as f:
            classifier = pickle.load(f)
            
    for emb in embs:
        if feature_extr_method == 'ViT':
            sims = cosine_similarity([emb], text_embs.iloc[:, :-1].to_numpy())
            max_id = np.argmax(sims, axis=1)
            classes.append(text_embs.iloc[:, -1].to_numpy()[max_id][0])
        elif feature_extr_method == 'paddle':
            classes.append(classifier.predict([emb])[0])
            
    classes = np.asarray(classes)
    unique, counts = np.unique(classes, return_counts=True)
    
    fps = time.time() - start_time
    fps = round(imgs_len/fps, 3)
    
    return unique[np.argmax(counts)], fps

#@st.cache_data
def search_images(embs, metaclass, top_k, feature_extr_method, 
                  preprocces_type, augment_image, use_metaclass, use_pca):
    """
    Do similar images search<br>
    """
    print('apply_model')
    if image_retriever is None:
        return [], 'NA'
        
    start_time = time.time()
    imgs_len = len(embs)
    
    all_sims = []
    all_ids = []
    if feature_extr_method in ['ViT', 'paddle']:
        all_ids, all_sims = image_retriever.search_image_in_base(embs, metaclass, top_k)
    
    sorted_pairs = sorted(zip(all_ids, all_sims), key=lambda x: x[1], reverse=True)
    
    top_sims = []
    top_ids = []
    for p in sorted_pairs: # for all augmented images chose most similar images from base
        if p[0] not in top_ids:
            top_ids.append(p[0])       
            top_sims.append(p[1]) 
        if len(top_ids) >= top_k:
            break 
            
    fps = time.time() - start_time
    fps = round(imgs_len/fps, 3)
    
    return zip(top_ids, top_sims), fps

def get_gpu_memory():
    """
    Get info about GPU usage (not usuful for this app because models run on onnx)<br>
    """
    result = subprocess.check_output(
                                    [
                                        'nvidia-smi', '--query-gpu=memory.used',
                                        '--format=csv,nounits,noheader'
                                    ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]


def main():
    global image_retriever

    st.sidebar.title("Configurations")

    st.sidebar.markdown('''---''')

    feature_extr_method = st.sidebar.selectbox("Feature extractor method",
                                               ("ViT", "paddle")) #, "keypoints"))
                                               
    preprocces_type = 'keypoints'
    if feature_extr_method != 'keypoints':
        preprocces_type = st.sidebar.selectbox("How to preprocces image",
                                                ("padding", "rect crop"))

                                               
    augment_image = st.sidebar.checkbox("Augment input image", True)
    
    if feature_extr_method != 'keypoints':
        use_metaclass = st.sidebar.checkbox("Find metaclass before image search", True)
    
    top_k = st.sidebar.slider("Top K images to show", 1, 50, 5, 1)
    
    if feature_extr_method != 'keypoints':
        use_pca = st.sidebar.checkbox("Use PCA for faster search", False)
        
        if use_pca:
            f_after_pca = st.sidebar.slider("N features left after PCA", 1, 512, 128, 1)
    
    st.sidebar.markdown('''---''')
    if st.sidebar.button("Just Update Button At Sidebar"):
        st.rerun()

    
    # ----------------------------------------------------------------
    
    
    start_search = True
    st.title("YouScan CV Test Task")
    if st.button("Just Update Button At The Beginning"):
        st.rerun()
        
    st.markdown('''#### Database path''')
    with open(pathes_cache_path) as f:
        pathes = json.load(f)
        base_path = pathes['base_path']
    
    base_path = st.text_input("Put database folder with images full path", base_path)
    if os.path.exists(base_path):
        st.text(f'Path is exists')
    else:
        st.text(f'Path is not exists')
    
    pathes['base_path'] = base_path
    with open(pathes_cache_path, 'w') as f:
        json.dump(pathes, f)
        
    st.markdown('''#### Load image''')
    with open(pathes_cache_path) as f:
        pathes = json.load(f)
        image_path = pathes['image_path']
    
    test_image = st.text_input("Put image full path", image_path)

    pathes['image_path'] = test_image
    with open(pathes_cache_path, 'w') as f:
        json.dump(pathes, f)

    test_image_loaded, test_image_shape = load_image(test_image)
    if test_image_loaded is not None:
        st.text(f'Image was loaded, size: {test_image_shape}')
    else:
        st.text(f'Image was not loaded')
        start_search = False
    
    if test_image_shape[0] > test_image_shape[1]:
        st.image(test_image_loaded[:, :, ::-1], 
                 width=int(test_image_shape[1] * (300/test_image_shape[0])))
    else:
        st.image(test_image_loaded[:, :, ::-1], width=300)
        
    if st.button("Run Similarity Search"):
        try:
            image_retriever = initialize_image_retriever(feature_extr_method, preprocces_type)
            #st.markdown(f"**Model init in {} seconds**") 
        except Exception as e:
            st.text(e)
        
        if feature_extr_method != 'keypoints':
            if image_retriever and image_retriever.model is not None:
                st.markdown(f'**Model was loaded**')
            else:
                st.markdown(f'**Model was not loaded**')   
                start_search = False
                initialize_image_retriever.clear(feature_extr_method, preprocces_type)
            
        if image_retriever and image_retriever.base is not None:
            st.markdown(f'**Base was loaded**')
        else:
            st.markdown(f'**Base was not loaded**')   
            start_search = False
            initialize_image_retriever.clear(feature_extr_method, preprocces_type)
        
        if feature_extr_method != 'keypoints' and use_pca:
            pca_fps = image_retriever.make_pca(f_after_pca)  
            st.markdown(f"**PCA calculated in {pca_fps} seconds**")  
        else:
            image_retriever.apply_pca = False
        
        if start_search:
            aug_images = [test_image_loaded]
            
            if augment_image:
                aug_images, aug_fps = apply_augmantation(test_image_loaded)
                st.markdown(f"**Augmentation FPS: {aug_fps} **")
                if aug_images:
                    st.text(f'Augment images:')
                    im = visualize_augmantation(aug_images, 300, 1000)
                    st.image(im[:, :, ::-1], width=im.shape[1])
            
            if feature_extr_method != 'keypoints':        
                prepr_images, preprocess_fps = apply_preprocess(aug_images, preprocces_type)
                st.markdown(f"**Preprocess FPS: {preprocess_fps}  (processed {len(aug_images)} images)**")
            
            embeddings, model_fps = apply_model(prepr_images, feature_extr_method)
            st.markdown(f"**Model FPS: {model_fps}  (processed {len(prepr_images)} images)**")
            
            st.markdown(f"**Target Image**")
            if test_image_shape[0] > test_image_shape[1]:
                st.image(test_image_loaded[:, :, ::-1], 
                         width=int(test_image_shape[1] * (300/test_image_shape[0])))
            else:
                st.image(test_image_loaded[:, :, ::-1], width=300)
            
            
            metaclass = None
            base_masked = image_retriever.base
            if use_metaclass and feature_extr_method != 'keypoints':
                metaclass, classify_fps = find_metaclass(embeddings, feature_extr_method, 
                                                         preprocces_type)
                base_masked = image_retriever.base[(image_retriever.base.category == metaclass)]
                base_masked = base_masked.reset_index(drop=True)
                st.markdown(f"**Metaclass: {metaclass}**")
                st.markdown(f"FPS: {classify_fps}  (processed {len(embeddings)} embeddings)")
        
    
            search_res, search_fps = search_images(embeddings, metaclass, top_k, 
                                                   feature_extr_method, preprocces_type,
                                                   augment_image, use_metaclass, use_pca)
            st.markdown(f"**Search FPS: {search_fps}  (processed {len(embeddings)} embeddings)**")
        
            for i, res in enumerate(search_res):
                image_path = os.path.join(base_path, base_masked.filename[res[0]])
                image_loaded, image_shape = load_image(image_path)
                if image_loaded is None:
                    image_loaded, image_shape = np.zeros((300, 300, 3), 
                                                         dtype=np.uint8), (300, 300, 3)
                
                images_col, info_col = st.columns(2)
                with images_col:
                    if image_shape[0] > image_shape[1]:
                        st.image(image_loaded[:, :, ::-1], 
                                 width=int(image_shape[1] * (300/image_shape[0])))
                    else:
                        st.image(image_loaded[:, :, ::-1], width=300)
                with info_col:
                    st.markdown(f'Similarity Score: **{res[1]}**')
                    if feature_extr_method != 'keypoints':
                        st.markdown(f'Metaclass: **{base_masked.category[res[0]]}**')
                    st.markdown(f'Image name: **{base_masked.filename[res[0]]}**')
                    
    
    
    st.markdown('''---''')
    st.subheader("System Stats")
    st1, st2, st3 = st.columns(3)

    with st1:
        st.markdown(f"**RAM Memory usage: {psutil.virtual_memory()[2]} %**")

    with st2:
        st.markdown(f"**CPU Usage: {psutil.cpu_percent()} %**")

    with st3:
        try:
            st.markdown(f"**GPU Memory Usage: {get_gpu_memory()} MB**")
        except:
            st.markdown(f"**GPU Memory Usage: NA**")

    if st.button("Just Update Button At The End"):
        st.rerun()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        print("System exited")
