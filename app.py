import glob
import streamlit as st
from PIL import Image
import cv2
import os
import time
import torch
from ultralytics import YOLO
from collections import Counter
import numpy as np

st.set_page_config(layout="wide")

cfg_model_path = 'models/best 908.pt'
model = None
confidence = .25


def image_input(data_src):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a test image.", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Selected Image")
        with col2:
            img, result = infer_image(Image.open(img_file))
            counter = quantity_estimate(result)
            st.image(img, caption="Model prediction")
            for k, v in counter.items():
                st.write(f"{v} {result[0].names[k]}")


def video_input(data_src):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "data/sample_videos/sample.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi', 'mkv'])
        if vid_bytes:
            vid_file = "data/uploaded_data/upload." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        fps = 0
        st1, st2, st3 = st.columns(3)
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty()
        prev_time = 0
        curr_time = 0
        stop = st.button("Stop")
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.resize(frame, (width, height))
            output_img, _ = infer_image(frame)
            output.image(output_img)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")
            if stop:
                cap.release()
                break

        cap.release()

def live_input():
    cap = cv2.VideoCapture(0)
    custom_size = st.sidebar.checkbox("Custom frame size")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if custom_size:
        width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
        height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

    fps = 0
    st1, st2, st3 = st.columns(3)
    with st1:
        st.markdown("## Height")
        st1_text = st.markdown(f"{height}")
    with st2:
        st.markdown("## Width")
        st2_text = st.markdown(f"{width}")
    with st3:
        st.markdown("## FPS")
        st3_text = st.markdown(f"{fps}")

    st.markdown("---")
    output = st.empty()
    prev_time = 0
    curr_time = 0
    stop = st.button("Stop")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Can't read frame, stream ended? Exiting ....")
            break
        frame = cv2.resize(frame, (width, height))
        output_img, _ = infer_image(frame)
        output.image(output_img)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        st1_text.markdown(f"**{height}**")
        st2_text.markdown(f"**{width}**")
        st3_text.markdown(f"**{fps:.2f}**")
        if stop:
            cap.release()
            break

    cap.release()


def infer_image(frame, size=None):
    results = model.predict(source=frame, show=False, conf=confidence, save=False)
    for r in results:
        im_array = r.plot()  
        im = Image.fromarray(im_array[..., ::-1])  
    return im, results

def quantity_estimate(result):    
    counter = Counter(result[0].boxes.cls.numpy().astype(int))
    return counter


@st.cache_resource
def load_model(cfg_model_path, device):
    model_ = YOLO(cfg_model_path)
    model_.to(device)
    print("model to ", device)
    return model_

def main(): 
    # global variables
    global model, confidence, cfg_model_path

    st.title("Object Recognition Dashboard")

    st.sidebar.title("Settings")

    # check if model file is available
    if not os.path.isfile(cfg_model_path):
        st.warning("Model file not available!!!, please add to the model folder.", icon="⚠️")
    else:
        # device options
        if torch.cuda.is_available():
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
        else:
            device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

        # load model
        model = load_model(cfg_model_path, device_option)

        # confidence slider
        confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

        model.classes = list(model.names.keys())

        st.sidebar.markdown("---")

        # input options
        input_option = st.sidebar.radio("Select input type: ", ['image', 'video', 'live'])

        # input src option
        if input_option is not 'live':
            data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'])

        if input_option == 'image':
            image_input(data_src)
        elif input_option == 'video':
            video_input(data_src)
        else:
            live_input()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass