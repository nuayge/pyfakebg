from tfjs_graph_converter import util as tfjs_util
from tfjs_graph_converter import api as tfjs_api
import tensorflow as tf
import numpy as np
import cv2
import os
import streamlit as st
import pyfakewebcam
import glob
from pyfakebg import constants
import time

V4L2_WIDTH, V4L2_HEIGHT = 640, 480


# make tensorflow stop spamming messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

@st.cache(hash_funcs={pyfakewebcam.v4l2.v4l2_format: id}, allow_output_mutation=True)
def open_fake_device():
    return pyfakewebcam.FakeWebcam("/dev/video2", V4L2_WIDTH, V4L2_HEIGHT)


#######################
# MobileNet preprocess:  RGB_Image / 127.5 - 1
# https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/mobilenet.ts#L24

# For ResNet, need to `np.add(frame, np.array([-123.15, -115.90, -103.06])`
# See https://github.com/tensorflow/tfjs-models/blob/master/body-pix/src/resnet.ts#L26
#######################

#######################
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py#L230
# Output Stride must be in [8,16,32]
#######################

def process_frame(frame, sess, graph, inputs, outputs, threshold, output_stride, model):
    """ Takes a BGR image and returns a segmentation mask corresponding to "where someone is"

    Parameters
    ----------
    frame : numpy array
        BGR frame
    sess : TensorFlow session
    graph : TensorFlow graph
    inputs : List
        graph's inputs
    outputs : List
        graph's outputs
    threshold : float
        float between 0. and 1. to create the segmentation mask.
        After applying a sigmoid to the segmentation output of the model,
        values below threshold will be 0, values above will be 1, giving a binary array
    output_stride : int
        (==8, 16 or 32)
        The output stride is inherent to to the chosen model,
        the higher the stride, the lower the resolution and faster the prediction
        MobileNet offers output strides of 8 and 16, ResNet 16 and 32
    model : str
        'mobilenet' or 'resnet'

    Returns
    -------
    numpy array
        A binary mask depending if someone is present at location (x,y)
    """

    height, width = frame.shape[:2]

    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    target_width = (int(processed_frame.shape[1]) // output_stride) * output_stride + 1
    target_height = (int(processed_frame.shape[0]) // output_stride) * output_stride + 1
    cv_sized_frame = cv2.resize(processed_frame, (target_height, target_width)).astype(np.float32)
    if model=='mobilenet':
        cv_sized_frame = cv_sized_frame / 127.5 - 1
    elif model=='resnet':
        np.add(processed_frame, [-123.15, -115.90, -103.06])
    ready_for_nn = cv_sized_frame[tf.newaxis, ...]
    input_tensor = graph.get_tensor_by_name(inputs[0])

    # Mobile and Res nets are not in the same order, ziping it with the
    # outputs names ensures we're looking at the right thing
    results = dict(zip(outputs, sess.run(outputs, feed_dict={
                           input_tensor: ready_for_nn})
                          )
                      )

    float_segments = results['float_segments:0'][0]
    sigmo = tf.sigmoid(float_segments)

    mask = tf.math.greater(sigmo, tf.constant(threshold))
    new_mask = tf.dtypes.cast(mask, np.uint8)

    return new_mask.numpy()


def capture_display_video(vc, sess, graph, nb_frames, fake_camera):
    i = 0
    input_tensor_names = tfjs_util.get_input_tensors(graph)
    output_tensor_names = tfjs_util.get_output_tensors(graph)

    ret, original_frame = vc.read()

    overlay = np.zeros(original_frame.shape)

    background = (
        cv2.resize(cv2.imread(chosen_background), (original_frame.shape[1], original_frame.shape[0]))
        if chosen_background and chosen_background is not "blur"
        else original_frame
    )

    while ret:
        if i % nb_frames == 0 and chosen_background:
            start = time.time()
            if chosen_background == "blur":
                background = cv2.GaussianBlur(original_frame, (blurring_kernel_size, blurring_kernel_size), 0)

            overlay = process_frame(
                original_frame,
                sess,
                graph,
                input_tensor_names,
                output_tensor_names,
                threshold,
                output_stride,
                model,
            )

            processing_time = time.time() - start
            start = time.time()

            overlay = cv2.resize(
                overlay, (original_frame.shape[1], original_frame.shape[0]), interpolation=cv2.INTER_CUBIC
            )
            overlay = np.reshape(overlay, (*overlay.shape, 1))
            second_text.text(f"Model took: {str(processing_time)[:5]}")

        displayed_frame = np.where(overlay==[0], background, original_frame) if chosen_background else original_frame

        if show_preview:
            placeholder.image(displayed_frame, channels="BGR")
        if write_to_device:
            fake_camera.schedule_frame(cv2.cvtColor(displayed_frame, cv2.COLOR_BGR2RGB))

        ret, original_frame = vc.read()
        i += 1

        if i == nb_frames+1:
            i = 0

    vc.release()


def main():
    global placeholder
    global text_placeholder
    global second_text
    global threshold
    global blurring_kernel_size
    global output_stride
    global chosen_background
    global chosen_heatmaps
    global chosen_type_output
    global model
    global write_to_device
    global show_preview

    # ResNet on CPU won't be fast
    fake_device_exists = False

    try:
        fake_camera = open_fake_device()
        fake_device_exists = True
    except Exception as e:
        st.error(f'Could not find fake device. {e}')
        st.info('Use `sudo modprobe v4l2loopback exclusive_caps=1`')

    model = st.sidebar.selectbox("Model", ["mobilenet", "resnet"])

    # Lower stride --> Higher res but slower
    output_stride = st.sidebar.selectbox("Output Stride", [8, 16], 1)
    if model == "mobilenet":
        model_path = f"models/bodypix_mobilenet_float_050_model-stride{output_stride}/"
    elif model == "resnet":
        model_path = f"models/bodypix_resnet50_float_model-stride{output_stride}"

    potential_backgrounds = [im for im in glob.glob("images/*")]

    nb_frames = st.sidebar.slider("Run Model every N frames", 1, 30, 3, 1)
    threshold = st.sidebar.slider("Sigmoid threshold", 0.0, 1.0, 0.7, 0.05)
    # https://github.com/streamlit/streamlit/issues/745 Can't use odd numbers in slider. 😿
    blurring_kernel_size = st.sidebar.slider("Kernel Size for blurring", 2, 100, 30, 2)
    blurring_kernel_size += 1
    chosen_type_output = st.sidebar.selectbox("Choose type of mask", ["Segmentation", "Heatmaps"])

    # TODO --> Choose parts we want
    # chosen_heatmaps = st.sidebar.multiselect(
    #     "Body parts heatmaps",
    #     [i for i, x in enumerate(constants.PART_CHANNELS)],
    #     format_func=lambda x: constants.PART_CHANNELS[x],
    # )

    chosen_background = st.sidebar.selectbox(
        "Background", ["", "blur"] + potential_backgrounds, format_func=lambda x: x.split("/")[-1]
    )

    show_preview = st.checkbox("Show preview ?", False)
    write_to_device = st.checkbox("Write to Device ?", False)
    placeholder = st.empty()
    text_placeholder = st.sidebar.empty()
    second_text = st.sidebar.empty()

    graph = tfjs_api.load_graph_model(model_path)
    sess = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1))

    if show_preview or write_to_device:
        vc = cv2.VideoCapture(0)
        vc.set(cv2.CAP_PROP_FPS, 1)
        if not vc.isOpened():
            raise Exception("Cannot capture video")
        capture_display_video(vc, sess, graph, nb_frames, fake_camera)

    sess.close()

if __name__ == "__main__":
    main()
