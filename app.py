import gc
import glob
import os
import time
from datetime import datetime
import utilsv8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
# import torch
from annotated_text import annotated_text
from PIL import Image
from streamlit_card import card
from streamlit_player import st_player
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import plotly.express as px
import cv2

import sys
sys.path.append('./ultralytics/yolo')

#### Yolov8 imports ####
from utilsv8 import get_detection_folderv8, check_folders
# import redirect as rd
# from ultralytics.yolo.engine.model import YOLO
from ultralytics import YOLO
from ultralytics.engine.results import Results

# from ultralytics.yolo.engine.results import Results

import streamlit as st
from PIL import Image
import os

# ### ResNeXt imports #####
# from dataset import ResNetDataset, classes, get_transforms
# from grad_cam import SaveFeatures, getCAM, plotGradCAM
# from html_mardown import (class0, class1,
#                           class2, class3, class4,
#                           image_uploaded_success, loading_bar,
#                           model_predicting, more_options, result_pred,
#                           s_load_bar, unknown, unknown_msg, unknown_side,
#                           unknown_w)
# from inference import inference, load_state
# from models import resnext50_32x4d
# from utilsgrad import CFG



logo = "./replant/tablogo.png"
st.set_page_config(page_title="Yolodetector", page_icon=logo, layout="centered", initial_sidebar_state="expanded")


COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
          (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100),
          (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]


def result_to_json(result: Results, tracker=None):
    """
    Convert result from ultralytics YOLOv8 prediction to json format
    Parameters:
        result: Results from ultralytics YOLOv8 prediction
        tracker: DeepSort tracker
    Returns:
        result_list_json: detection result in json format
    """
    len_results = len(result.boxes)
    result_list_json = [
        {
            'class_id': int(result.boxes.cls[idx]),
            'class': result.names[int(result.boxes.cls[idx])],
            'confidence': float(result.boxes.conf[idx]),
            'bbox': {
                'x_min': int(result.boxes.xyxy[idx][0]),
                'y_min': int(result.boxes.xyxy[idx][1]),
                'x_max': int(result.boxes.xyxy[idx][2]),
                'y_max': int(result.boxes.xyxy[idx][3]),
                # 'x_min': int(result.boxes.boxes[idx][0]),
                # 'y_min': int(result.boxes.boxes[idx][1]),
                
            },
        } for idx in range(len_results)
    ]
    if result.masks is not None:
        for idx in range(len_results):
            result_list_json[idx]['mask'] = cv2.resize(result.masks.data[idx].cpu().numpy(), (result.orig_shape[1], result.orig_shape[0])).tolist()
            result_list_json[idx]['segments'] = result.masks.segments[idx].tolist()
    if tracker is not None:
        bbs = [
            (
                [
                    result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_min'],
                    result_list_json[idx]['bbox']['x_max'] - result_list_json[idx]['bbox']['x_min'],
                    result_list_json[idx]['bbox']['y_max'] - result_list_json[idx]['bbox']['y_min']
                ],
                result_list_json[idx]['confidence'],
                result_list_json[idx]['class'],
            ) for idx in range(len_results)
        ]
        
    return result_list_json


def view_result_default(result: Results, result_list_json, centers=None):
    ALPHA = 0.5
    image = result.orig_img
    hprinted = False
    bprinted = False
    pprinted = False
    lprinted = False
    nprinted = False
    for result in result_list_json:
        class_color = COLORS[result['class_id'] % len(COLORS)]
        if 'mask' in result:
            image_mask = np.stack([np.array(result['mask']) * class_color[0], np.array(result['mask']) * class_color[1], np.array(result['mask']) * class_color[2]], axis=-1).astype(np.uint8)
            image = cv2.addWeighted(image, 1, image_mask, ALPHA, 0)
        text = f"{result['class']} {result['object_id']}: {result['confidence']:.2f}" if 'object_id' in result else f"{result['class']}: {result['confidence']:.2f}"
        # st.write(text)
        # st.write(result['class'])
        


        if result['class'] == "Healthy" and not hprinted:
            label_name = "Healthy"
            conf = result['confidence']*100
            st.success(f'✅ The Predicted Class is :  "{label_name}" {conf:.2f} % ')

            st.write("")
            st.image("./replant/healthy.png")
            st.markdown("***")
            hprinted = True
        
        if result['class'] == "Leaf_Spot" and not lprinted:
            label_name = "Leaf_Spot"
            conf = result['confidence']*100
            st.success(f'✅ The Predicted Class is :  "{label_name}" {conf:.2f} % ')

            st.image("./replant/Spot1.png")
            st.markdown("***")

            st.image("./replant/SpotPM.png")
            st.image("./replant/SpotI.png")
            st.markdown("***")

            st.image("./replant/SpotCause.png")
            st.markdown("***")

            st.image("./replant/ProductRecom1.png")
            


            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)


            with col1:
                    #st.write('Caption for second chart')
                    hasClicked = card(
        title="GreenDrop",
        text="Enrich the soil and enhance beneficial microbes helps in nitrogen fixation.",
        image="https://plantic.in/image/greendrop.jpg",
        url="https://plantic.in/products/organic-greendrop?source=google&medium=cpc&campaignid=16514436694&adgroupid=&keyword=&matchtype=&device=c&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdphpQ-yJVyyxVQLz3Q8RBR0G8LC3Ai_5os9tlLhbu9gyfZqwwRSU6waAhWTEALw_wcB"
    )      
                
            with col2:
                hasClicked1 = card(
        title="Herbal Garden Protection",
        text="Herbal water based Eco-friendly spray made from Aromatic Oils and plant extracts.",
        image="https://cdn.shopify.com/s/files/1/0577/8951/3913/products/HBPA38-1_900x.jpg?v=1639130544",
        url="https://herbalstrategi.com/products/herbal-garden-protection-spray-for-pest-and-fungi-protection-500-ml-wellness-spray-bio-spray-for-faster-plant-growth-500-ml?variant=42162982387950&utm_source=google&utm_medium=cpc&utm_campaign=Google+Shopping&currency=INR&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdqBBniS-xFaHcbHe7VmYWn8BgGhbVHyhWLZrnUNfomdLia5rtfEwe8aAi8AEALw_wcB"
    )
            with col3:
                hasClicked2 = card(
        title="Trichoderma Viride",
        text="Excellent for suppressing diseases caused by fungal pathogens",
        image="https://cdn.shopify.com/s/files/1/0451/1101/7626/products/33_d58110d7-1c92-4b8c-93ab-ed17a122f914_800x.png?v=1678039208",
        url="https://seed2plant.in/products/trichoderma-viride?currency=INR&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdogp8XpyWqEyWWHsdC9xtNnNlsqyJE12vULNvKn1KEblURtxwFdfMcaAo5IEALw_wcB"
    )
            with col4:
                hasClicked3 = card(
        title="NPK Fertilizer",
        text="It controls leaf yellowing, improves green leaves and prevent the Black spots",
        image="https://m.media-amazon.com/images/I/71JcFuVYRFL._SX522_.jpg",
        url="https://www.amazon.in/19-Fertilizer-Garden-Plants-0-25/dp/B0B1MZV3DS/ref=asc_df_B0B1MZV3DS/?tag=googleshopdes-21&linkCode=df0&hvadid=586198977745&hvpos=&hvnetw=g&hvrand=10187752443105046934&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1007810&hvtargid=pla-1722813852198&th=1"
    )
            st.markdown("***")
            
            st.image("./replant/Careguideyoutube.png") 
            yt1, yt2 = st.columns([1, 1])
            with yt1:
            # Embed a youtube video
                st_player("https://youtu.be/dnNJwQ86c4w")

            with yt2:
            # Embed a music from SoundCloud
                st_player("https://youtu.be/NcnHd4xSMvk")
            lprinted = True

             
        if result['class'] == "Blight" and not bprinted:
            label_name = "Blight"
            conf = result['confidence']*100
            st.success(f'✅ The Predicted Class is :  "{label_name}" {conf:.2f} % ')


            recommend = "- Prune the diseased leaves. Including weed control in the planting area. to reduce the accumulation of pathogens"
            treatment = "- Cut out the diseased branches, burn them. (If it is a large branch, it should be applied with red lime or copper compounds) and then sprayed with carbendashim. (carbendazim) 60% WP rate 10 grams per 20 liters of water or copper oxychloride 85% WP rate 50 grams per 20 liters of water throughout the interior and exterior."
            
            st.image("./replant/Blight1.png")
            st.markdown("***")

            st.image("./replant/BlightPM.png")
            st.image("./replant/BlightI.png")
            st.markdown("***")

            st.image("./replant/BlightCause.png")
            st.markdown("***")

            st.image("./replant/BlightHM.png")
            st.markdown("***")

            st.image("./replant/ProductRecom1.png")
        
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)


            with col1:
                    #st.write('Caption for second chart')
                    hasClicked = card(
        title="Kavach Fungicide",
        text="Contains Chlorothalonil is a broad-spectrum contact fungicide and is highly effective against Anthracnose, Fruit Rots, Early and Late Blight on various crops.",
        image="https://cdn.shopify.com/s/files/1/0722/2059/products/3_35_800x.webp?v=1672228385",
        url="https://www.bighaat.com/products/kavach-fungicide?variant=31592941977623&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=Google&utm_medium=CPC&utm_campaign=17706716593&utm_adgroup=&utm_term=&creative=&device=c&devicemodel=&matchtype=&feeditemid=&targetid=&network=x&placement=&adposition=&GA_loc_interest_ms=&GA_loc_physical_ms=1007810&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdqiXU6Un3-UNUfDmbDkNTItG7qie77l235Xk5ANEddTTVFiPuUpH9AaAs0XEALw_wcB"
    )      
                
            with col2:
                hasClicked1 = card(
        title="MYCICON",
        text="Fungicide For Controlling Fungal Infection Like Powdery, Downy Mildew And Blight",
        image="https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcQ1AD4mlIQHnyau2MjSmqOXEKv4GwUuATWcMquOM4GF94nQz7LZcTsj_23qOUzecP7Pf6-rEz2r1BBEZ2kIH0TsX30CNC6CzCT1lV2WFHnGwcmBUfZCW56dog&usqp=CAc",
        url="https://agribegri.com/products/svfvgf.php"
    )
            with col3:
                hasClicked2 = card(
        title="Green-Drop",
        text="Enrich the soil and enhance beneficial microbes helps in nitrogen fixation.",
        image="https://plantic.in/image/greendrop.jpg",
        url="https://plantic.in/products/organic-greendrop?source=google&medium=cpc&campaignid=16514436694&adgroupid=&keyword=&matchtype=&device=c&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdphpQ-yJVyyxVQLz3Q8RBR0G8LC3Ai_5os9tlLhbu9gyfZqwwRSU6waAhWTEALw_wcB"
    )      
            with col4:
                hasClicked3 = card(
        title="NPK 19-19-19 Fertilizer",
        text="It controls leaf yellowing, improves green leaves and prevent the Black spots",
        image="https://m.media-amazon.com/images/I/71JcFuVYRFL._SX522_.jpg",
        url="https://www.amazon.in/19-Fertilizer-Garden-Plants-0-25/dp/B0B1MZV3DS/ref=asc_df_B0B1MZV3DS/?tag=googleshopdes-21&linkCode=df0&hvadid=586198977745&hvpos=&hvnetw=g&hvrand=10187752443105046934&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1007810&hvtargid=pla-1722813852198&th=1"

    )
            st.markdown("***")
            st.image("./replant/Careguideyoutube.png")   
            ytb1, ytb2 = st.columns([1, 1])
            with ytb1:
            # Embed a youtube video
                st_player("https://youtu.be/cGhzyhnKi5U")

            with ytb2:
            # Embed a music from SoundCloud
                st_player("https://youtu.be/eTA8VFeE-6Q")
            bprinted = True

    
        
        if result['class'] == "Nitrogen_Deficiency" and not nprinted:
            label_name = "Nitrogen Deficiency Symptoms"
            conf = result['confidence']*100
            st.success(f'✅ The Predicted Class is :  "{label_name}" {conf:.2f} % ')

            recommend = "None"
            treatment = "- Soil fertilization: Mix NPK fertilizer with the highest N ratio and observe the amount of application according to the symptoms of the leaves. \n - Foliar fertilization: Use chemical fertilizers with high N values ​​or use urea. Swimming, high nitrogen formula Mix and get out of the water fertilizer."
            st.image("./replant/N1.png")
            st.image("./replant/NI.png")
            st.markdown("***")

            st.image("./replant/NPM.png")
            st.markdown("***")

            st.image("./replant/ProductRecom1.png")
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                    #st.write('Caption for second chart')
                    hasClicked = card(
        title="RAW- Nitrogen, Plant Nutrient",
        text="For treating deficiencies, increase plant growth during vegative Stage, plant feeding supplement, for Indoor Outdoor Use",
        image="https://m.media-amazon.com/images/I/61zSF3cO7HL._AC_SX522_.jpg",
        url="https://www.amazon.com/NPK-Industries-717891-RAW-Nitrogen/dp/B00UL3YGWO?th=1"
    )      
                
            with col2:
                hasClicked1 = card(
        title="AquaNature Flora-N",
        text="High quality fertilizer for freshwater planted aquariums",
        image="http://www.aquanatureonline.com/wp-content/uploads/2021/08/FLORA-N.jpg",
        url="http://www.aquanatureonline.com/product/aquanature-flora-n-concentrated-nitrogen-supplement-for-planted-aquarium/?attribute_pa_size-upto=250ml"
    )
            
            with col3:
                    hasClicked22 = card(
        title="Organic Blood Meal",
        text="Provides essential micro-nutrients for plants growth",
        image="https://m.media-amazon.com/images/I/71s669Kke3L._SX679_.jpg",
        url="https://www.amazon.in/BreathingLeaf-Organic-Gardening-Excellent-Nitrogen/dp/B08T62XGQ9"
    )      
                
            with col4:
                hasClicked33 = card(
        title="Miracle-Gro",
        text="Specially formulated food for indoor plants",
        image="https://m.media-amazon.com/images/I/610FuWdoXLL._AC_SX679_.jpg",
        url="https://www.amazon.com/Miracle-Gro-100055-Indoor-Plant-1-1-1/dp/B0071E21ZU/ref=sr_1_5?keywords=Nitrogen%2Bfor%2BPlants&qid=1677964343&sr=8-5&th=1"
    )
                
            st.markdown("***")
                
            st.image("./replant/Careguideyoutube.png")
            ytn1, ytn2 = st.columns([1, 1])
            with ytn1:
            # Embed a youtube video
                st_player("https://youtu.be/-g4TSPkVJUU")

            with ytn2:
            # Embed a music from SoundCloud
                st_player("https://youtu.be/vdSTlA_FtbY")   
            nprinted = True


        if result['class'] == "Powdery_Mildew" and not pprinted:
            label_name = "Powdery Mildew"
            conf = result['confidence']*100
            st.success(f'✅ The Predicted Class is :  "{label_name}" {conf:.2f} % ')

            recommend = "🌱 If a plant displays signs of a disease, remove all the infected parts and destroy them by burning.\n 🌱 Replace the soil in the flowerpot.\n 🌱 Use only settled room-temperature water forwatering.\n 🌱 Adjust the air conditions. If houseplants are infected,keep them spaced."
            treatment = "- After harvesting the crop To destroy the plant debris that used to be diseased by tilling. And crop rotation. \n - Spraying fungicides such as triadimefon, myclobutanil. (myclobutanil) propiconazole (propiconazole) azocystrobin (azoxystrobin)"
            st.image("./replant/Powdery1.png")
            st.markdown("***")

            st.image("./replant/PowderyPM.png")
            st.image("./replant/PowderyI.png")
            st.markdown("***")

            st.image("./replant/PowderyHM.png")
            st.markdown("***")

            st.image("./replant/ProductRecom1.png")


            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)


            with col1:
                   #st.write('Caption for second chart')
                   hasClicked = card(
        title="Powdery Prochloraz",
        text="Midazole fungicide that is widely used in gardening and agriculture",
        image="https://cdn.shopify.com/s/files/1/0722/2059/products/4copy_1800x1800.webp?v=1672229156",
        url="https://www.bighaat.com/products/score-fungicide?variant=12725272936471&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=Google&utm_medium=CPC&utm_campaign=17706716593&utm_adgroup=&utm_term=&creative=&device=c&devicemodel=&matchtype=&feeditemid=&targetid=&network=x&placement=&adposition=&GA_loc_interest_ms=&GA_loc_physical_ms=1007810&gclid=Cj0KCQiAofieBhDXARIsAHTTldpXkjTg0o32bEGopU2HNKUUZVseCAvqWfX6tgApx_MFEWtPNGi8cu4aAjLhEALw_wcB"
    )      

            with col2:
                hasClicked1 = card(
        title="Patch Pro Fungicide",
        text="Systemic fungicide that contains the active ingredient Propiconazole",
        image="https://smhttp-ssl-60515.nexcesscdn.net/media/catalog/product/cache/1/image/600x600/9df78eab33525d08d6e5fb8d27136e95/p/a/patch_pro_grn_shadow2_1.jpeg",
        url="https://www.solutionsstores.com/patch-pro-fungicide"
    )
            with col3:
                hasClicked2 = card(
        title="SAAF Fungicide",
        text="Controls Anthracnose, Powdery mildew AND Rust Disease",
        image="https://cdn.shopify.com/s/files/1/0722/2059/products/Saaf_1800x1800.webp?v=1680086906",
        url="https://www.bighaat.com/products/upl-saaf-fungicide?variant=31478554722327&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=Google&utm_medium=CPC&utm_campaign=16667009224&utm_adgroup=&utm_term=&creative=&device=c&devicemodel=&matchtype=&feeditemid=&targetid=&network=x&placement=&adposition=&GA_loc_interest_ms=&GA_loc_physical_ms=1007810&gclid=Cj0KCQiA54KfBhCKARIsAJzSrdoz8vri-PqBXgXRx7JCt1TEZFVXPtt4PRoj_KxcRXxc4xCzhrKmc9saAuhFEALw_wcB"
    )
            with col4:
                hasClicked3 = card(
        title="Leemark",
        text="Prevent Powdery mildew, Black spot, Downy mildew, Blights & Molds",
        image="https://badikheti-production.s3.ap-south-1.amazonaws.com/products/202301281242101383789267.jpg",
        url="https://www.badikheti.com/organic-pesticide/pdp/leemark-prevent-powdery-mildew-black-spot-downy-mildew-blights-molds-and-other-plant-diseases/269unrka"
    )
            st.markdown("***")
            st.image("./replant/Careguideyoutube.png")

            youtubec1, youtubec2 = st.columns([1, 1])
            with youtubec1:
            # Embed a youtube video
                st_player("https://youtu.be/Kzm6jxeU1kg")

            with youtubec2:
            # Embed a music from SoundCloud
                st_player("https://youtu.be/xEqiNh0upMk")  
            pprinted = True


        if result['class'] == " ": # No disease found in the picture.
            st.warning("No disease found in the picture !! Please take a New photo")   
    


        cv2.rectangle(image, (result['bbox']['x_min'], result['bbox']['y_min']), (result['bbox']['x_max'], result['bbox']['y_max']), class_color, 3)
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 3)
        cv2.rectangle(image, (result['bbox']['x_min'], result['bbox']['y_min'] - text_height - baseline), (result['bbox']['x_min'] + text_width, result['bbox']['y_min']), class_color, -1)
        cv2.putText(image, text , (result['bbox']['x_min'], result['bbox']['y_min'] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if 'object_id' in result and centers is not None:
            centers[result['object_id']].append((int((result['bbox']['x_min'] + result['bbox']['x_max']) / 2), int((result['bbox']['y_min'] + result['bbox']['y_max']) / 2)))
            for j in range(1, len(centers[result['object_id']])):
                if centers[result['object_id']][j - 1] is None or centers[result['object_id']][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image, centers[result['object_id']][j - 1], centers[result['object_id']][j], class_color, thickness)
        

        

        #st.image(image, channels="BGR")
    return image



def image_processing(frame, model, image_viewer=view_result_default, tracker=None, centers=None):
    results = model.predict(frame)
    result_list_json = result_to_json(results[0], tracker=tracker)
    result_image = image_viewer(results[0], result_list_json, centers=centers)
    return result_image, result_list_json



def imageInput(device):
    image_file = st.file_uploader(label = "Upload image of the leaf here.. ",type=['png','jpg','jpeg'])
    if image_file is not None:

        st.caption("### Detection result.. (Summary of the analysis)")
        

        #img = Image.open(image_file)
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
        outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
        #st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
        st.sidebar.image(image_file, width=301, channels="BGR")
                
        with open(imgpath, mode="wb") as f:
            f.write(image_file.getbuffer())
            salie_im = cv2.imread(imgpath)
            annotated_text(("Saliency mask view","","#2E7C30"))
            st.image(salie_im, width=528, channels="RGB")

        # call Model prediction--
       
        #  model = torch.hub.load('ultralytics/yolov5', 'custom' , path ='models/yleafinev5.pt', force_reload = True, _verbose = False)
        
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'models/yleafinev5.pt', force_reload = True, _verbose = False)
        # _ = model.cuda() if device == 'cuda' else model.cpu() # hide cuda_cnn display source : https://stackoverflow.com/questions/41149781/how-to-prevent-f-write-to-output-the-number-of-characters-written
        # pred = model(imgpath)
        # st.write(pred)
        # pred.render()  # render bbox in image
        # for im in pred.ims:
        #     im_base64 = Image.fromarray(im)
        #     im_base64.save(outputpath)

        # pred.save()
        # detect_val = (pred.pandas().xyxy[0]).values.tolist()

      
        #to know the detection results
        #st.write(detect_val)

        annotated_text(("YOLOv8","detections","#2E7C30"))
        out1, out2 = st.columns(2)
       
         # --Display predicton / print result
        with out2: 
            # img_out = Image.open(outputpath)
            # annotated_text(("YOLOv5","detections","#1F617C"))
            st.write("")
            # st.image(img_out)
        
        st.markdown("***")
        #get_accuracy_str(detect_val) # get detection string result
        
        # st.write(subprocess.run(['yolo', 'task=detect', 'mode=predict', 'model=v8.pt', 'conf=0.45', 'line_thickness=1', 'source={}'.format(source)],capture_output=True, universal_newlines=True,).stderr)
        check_folders()
        
        # YOLOV8 DETECTION RESULTS
        
        model1 = YOLO(f'models/v8.pt')
        #with st.spinner("Detecting with 💕"):
        if image_file is not None:
            st.write("")
            st.sidebar.success("Successfully uploaded")
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
            ## for detection with bb
            print(f"Used Custom reframed YOLOv8 model: {model1}")
            img, result_list_json = image_processing(img, model1)
              # out2 = st.columns(1)
            with out1:
              with st.spinner("Detecting with 💕"):
                  # print(json.dumps(result_list_json, indent=2))
                  st.write(" ")
                  st.image(img, channels="BGR")
                      
            
            
        #st.write(results)
                
        # st.balloons()

        
       
       ### ReDIRECT For Manure Suggestions..

    else:
        st.caption("")
        # st.image("./replant/tips.png")
        # st.image("./replant/tip1.png")
         
         

def main(): 
    #Logo image here
    #st.sidebar.image("./replant/logoleafine.png")
    #st.sidebar.caption("### Your Plant Wellbeing Assistant🍃 ###") 
    #st.sidebar.title('⚙️ Select option')
    # activities = ["Detection with YOLO (Analyzed disease)", "Detection with ResNeXt (GradCam Visualization)"]
    # choice = st.sidebar.selectbox("# Click here to know more.. #",activities)

    #st.sidebar.markdown("https://bit.ly/3uvYQ3R")

    #background image
       
        st.markdown("<h1 style='text-align: left; color: white; text-shadow: 5px 5px 5px green; font-size: 60px; font-family: 'Trebuchet MS', sans-serif;'>Leaf Disease Detector</h1>", unsafe_allow_html=True)

        st.text("")
        st.text("")


        # st.image("./replant/logoleafine.png")
        # Perceive the leaf ailment and sort out some way to treat them
        st.caption('### Recognize & Perceive the leaf illness and figure out how to treat them!')
        st.markdown("***")
        
        col1, col2 = st.columns(2)

        # with col2:
        #     # option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
        #     if torch.cuda.is_available():
        #         deviceoption = st.radio("Select runtime mode :", ['cpu', 'cuda (GPU)'], index=1)
        #     else:
        #         deviceoption = st.radio("Select runtime mode :", ['cpu', 'cuda (GPU)'], index=0)
        #     # -- End of Sidebar
        # with col1:
        #     pages_name = ['Upload own data']
        #     page = st.radio('Select option mode :', pages_name) 
        
        page = 'upload own data'

        # if page == "Upload own data":
        # st.subheader('🔽Upload Image 📸')
        t1 = time.perf_counter()
        deviceoption = 'cpu'
        # deviceoption = 'cpu'
        imageInput(deviceoption)
        t2 = time.perf_counter()
        st.success('Time taken to run: {:.2f} sec'.format(t2-t1))

      

    
    # elif choice == 'Detection with ResNeXt (GradCam Visualization)' :

    #     # Enable garbage collection
    #     gc.enable()

    #     # Hide warnings
    #     st.set_option("deprecation.showfileUploaderEncoding", False)

    #     # Set the directory path
    #     my_path = "."

    #     test = pd.read_csv("data/sample.csv")
    #     output_image = my_path + "/images/gradcam2.png"
    #     st.image("./replant/logoleafine.png")
    #     # Perceive the leaf ailment and sort out some way to treat them
    #     st.caption('### GradCam Visualization and Detection with ResNeXt Detector ###')
        
    #     st.markdown("***")

        # def run():
        # # Set the box for the user to upload an image
        # st.subheader('🔽Upload Image 📸')
        # uploaded_image = st.file_uploader(
        #         "Upload your image in JPG or PNG format", type=["jpg", "png"]
        #     )
        uploaded_image=None
        #     return uploaded_image
        # st.image("./replant/tips.png")
        # st.image("./replant/tip2.png")


        # DataLoader for pytorch dataset
        def Loader(img_path=None, uploaded_image=None, upload_state=False, demo_state=True):
            test_dataset = ResNetDataset(
                test,
                img_path,
                uploaded_image=uploaded_image,
                transform=get_transforms(data="valid"),
                uploaded_state=upload_state,
                demo_state=demo_state,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=CFG.batch_size,
                shuffle=False,
                num_workers=CFG.num_workers,
                pin_memory=True,
            )
            return test_loader


        # Function to deploy the model and print the report
        def deploy(file_path=None, uploaded_image= uploaded_image, uploaded=False, demo=True):
            # Load the model and the weights
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = resnext50_32x4d(CFG.model_name, pretrained=False)
            states = [load_state("models/resnext50_32x4d_fold0_best.pth")]

            # For Grad-cam features
            final_conv = model.model.layer4[2]._modules.get("conv3")
            fc_params = list(model.model._modules.get("fc").parameters())

            # Display the uploaded/selected image
            st.markdown("***")
            st.markdown(model_predicting, unsafe_allow_html=True)
            if demo:
                test_loader = Loader(img_path=file_path)
                image_1 = cv2.imread(file_path)
            if uploaded:
                test_loader = Loader(
                    uploaded_image=uploaded_image, upload_state=True, demo_state=False
                )
                image_1 = file_path
            st.sidebar.markdown(image_uploaded_success, unsafe_allow_html=True)
            st.sidebar.image(image_1, width=301, channels="BGR")

            for img in test_loader:
                activated_features = SaveFeatures(final_conv)
                # Save weight from fc
                weight = np.squeeze(fc_params[0].cpu().data.numpy())

                # Inference
                logits, output = inference(model, states, img, device)
                pred_idx = output.to("cpu").numpy().argmax(1)

                # Grad-cam heatmap display
                heatmap = getCAM(activated_features.features, weight, pred_idx)

                ##Reverse the pytorch normalization
                MEAN = torch.tensor([0.485, 0.456, 0.406])
                STD = torch.tensor([0.229, 0.224, 0.225])
                image = img[0] * STD[:, None, None] + MEAN[:, None, None]

                # Display image + heatmap
                plt.imshow(image.permute(1, 2, 0))
                plt.imshow(
                    cv2.resize(
                        (heatmap * 255).astype("uint8"),
                        (328, 328),
                        interpolation=cv2.INTER_LINEAR,
                    ),
                    alpha=0.4,
                    cmap="jet",
                )
                plt.savefig(output_image)

                # Display Unknown class if the highest probability is lower than 0.5
                if np.amax(logits) < 0.57:
                    st.markdown(unknown, unsafe_allow_html=True)
                  #  st.sidebar.markdown(unknown_side, unsafe_allow_html=True)
                   # st.sidebar.markdown(unknown_w, unsafe_allow_html=True)

                # Display the class predicted if the highest probability is higher than 0.5
                else:
                    st.write("")
                    if pred_idx[0] == 0:
                       # st.markdown(class0, unsafe_allow_html=True)

                        st.success(" The predicted class is: **Bacterial Blight**")
                    elif pred_idx[0] == 1:
                        #st.markdown(class1, unsafe_allow_html=True)

                        st.success(
                            "The predicted class is: **Nitrogen Deficiency**"
                        )
                    elif pred_idx[0] == 2:
                       # st.markdown(class2, unsafe_allow_html=True)

                        st.success("The predicted class is: **Leaf Spot**")
                    elif pred_idx[0] == 3:
                        #st.markdown(class3, unsafe_allow_html=True)

                        st.success("The predicted class is: **Anthracnose**")
                    elif pred_idx[0] == 4:
                        #st.markdown(class4, unsafe_allow_html=True)

                        st.success("The predicted class is: **Healthy**")

                st.sidebar.markdown(
                    "**Scroll down to read the full report (Grad-cam and class probabilities)**"
                )

                # Display the Grad-Cam image
                st.caption("# Grad-cam visualization ##")
                st.caption("Grad-CAM (Gradient-weighted Class Activation Mapping) can be used to generate visualizations that highlight the regions of a leaf image that are most indicative of disease. " )
                st.write("")  
                st.caption("It assists with understanding if the model put together its predictions on the correct regions of the image.")
                gram_im = cv2.imread(output_image)
                #st.image(gram_im, width=528, channels="BGR")
                st.image("./images/gradcam2.png",width=560)

                # Display the class probabilities table
                st.caption("## Class Predictions: ##")
                st.write("")
                if np.amax(logits) < 0.57:
                    #st.markdown(unknown_msg, unsafe_allow_html=True)
                    st.write("")
                classes["CONFIDENCE📊%"] = logits.reshape(-1).tolist()
                classes["CONFIDENCE📊%"] = classes["CONFIDENCE📊%"] * 100
                cm = sns.color_palette("blend:white,green", as_cmap=True)
                classes_proba = classes.style.background_gradient(cmap=cm)
                st.write(classes_proba)

                fig = px.bar(x=classes['LABELS🏷️'], y=classes['CONFIDENCE📊%'],color=classes['CONFIDENCE📊%'],
                title="Bar plot which holds Labels on x-axis and Confidence of the disease on y-axis",)
                st.write(fig)
                
                del (
                    model,
                    states,
                    fc_params,
                    final_conv,
                    test_loader,
                    image_1,
                    activated_features,
                    weight,
                    heatmap,
                    gram_im,
                    logits,
                    output,
                    pred_idx,
                    classes_proba,
                )
                gc.collect()



        # Deploy the model if the user uploads an image
        if uploaded_image is not None:
            # Close the demo
            choice = "Select an Image"
            # Deploy the model with the uploaded image
            deploy(uploaded_image, uploaded=True, demo=False)
            del uploaded_image

#start
if __name__ == '__main__':
    main()

