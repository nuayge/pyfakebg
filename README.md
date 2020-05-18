# pyfakebg

This project aims at providing a "blurring background", a feature that is absent on most Linux implementation of teleconferencing softwares.

It runs pretty smoothly on CPU

Based on [BodyPix](https://github.com/tensorflow/tfjs-models/tree/master/body-pix), [v4l2loopback](https://github.com/umlaeute/v4l2loopback) and its Python3 binder [pyfakewebcam](https://github.com/jremmons/pyfakewebcam), [tfjs to tf converter](https://github.com/patlevin/tfjs-to-tf) and [Streamlit](https://github.com/streamlit/streamlit) to create a user-friendly interface.

Little example, working well with an uneven background.
![Example using Zoom](docs/demo_2.gif)


To use it:

First we need to create a virtual webcam.
Get v4l2loopback via
> `sudo apt install -y v4l2loopback-dkms`

Create fake device:
> `modprobe v4l2loopback exclusive_caps=1`

(The `exclusive_caps` here makes it usable in Chrome)

# Now:
> Clone this repo

From folder
> `pip install -r requirements.txt`

> `streamlit run interface.py`

> In your browser, go to localhost:8051


I included one model (MobileNet with stride 16 and float50, the smallest one)
You can download others by running 'get-models.sh' which I took somewhere I cannot remember (sorry for the author)

Just ctrl+F for 'bodypix' on https://storage.googleapis.com/tfjs-models to see which ones are available.

## Good to go.
You can have a preview in browser or directly write to the fake device previously created. Open Zoom, you'll have a "Dummy Device" which you can choose to display.
