import sys
from subprocess import call
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import string

def run_cmd(command):
    try:
        print(command)
        call(command, shell=True)
    except Exception as e:
        print(f"Errorrrrr: {e}!")


import cv2
import yaml

import matplotlib.pyplot as plt
from skimage import io
import gradio as gr


def inference_vid (video):
    run_cmd(f'python scripts/demo_video.py --out-dir res_dance --save-pk --video-name '+ video)
    return f'res_dance/res_'+ os.path.basename(video)

def inference_img (img):
    filename = os.path.basename(img)
    run_cmd(f'python scripts/demo_image.py --out-dir res --save-pk --img-single '+ img)
    return f'res/'+ filename

def inference_vid_x (video):
    run_cmd(f'python scripts/demo_video_x.py --out-dir res_dance --save-img --save-pk --video-name '+ video)
    return f'res_dance/res_'+ os.path.basename(video)

def inference_img_x (img):
    filename = os.path.basename(img)
    run_cmd(f'python scripts/demo_image_x.py --out-dir res --save-pk --img-single '+ img)
    return f'res/res_images/'+ filename

#video SMPL
title_vid = "HybrIK_vid"
description_vid = "HybrIK demo for gradio"
article_vid = "<p style='text-align: center'><a href='https://github.com/Jeff-sjtu/HybrIK/tree/main'>Github Repo</a></p>"
examples_vid = [   
['examples/dance.mp4'], 
]

vid_demo = gr.Interface(
     inference_vid,     
     gr.Video(), 
     outputs=gr.Video(label="Out"),
    title=title_vid,
    description=description_vid,
    article=article_vid,
    examples=examples_vid

)

#single image SMPL
title_img = "HybrIK_img"
description_img = "HybrIK image demo for gradio"
article_img = "<p style='text-align: center'><a href='https://github.com/Jeff-sjtu/HybrIK/tree/main'>Github Repo</a></p>"
examples_img = [   
    ['examples/000000000431.jpg'], 
    ['examples/000000581056.jpg'], 
    ['examples/000000581091.jpg'], 
    ['examples/000000581328.jpg'], 
    ['examples/000000581357.jpg'], 
    ['examples/000000581667.jpg']
]

img_demo = gr.Interface(
    inference_img,     
     gr.Image(type="filepath"), 
     outputs=gr.Image(type="filepath", label="Out"),
    title=title_img,
    description=description_img,
    article=article_img,
    examples=examples_img
    )

#image SMPL_X
title_img_x = "HybrIK_img_X"
description_img_x = "HybrIK_X demo for gradio"
article_img_x = "<p style='text-align: center'><a href='https://github.com/Jeff-sjtu/HybrIK/tree/main'>Github Repo</a></p>"
examples_img_x = [   
    ['examples/000000000431.jpg'], 
    ['examples/000000581056.jpg'], 
    ['examples/000000581091.jpg'], 
    ['examples/000000581328.jpg'], 
    ['examples/000000581357.jpg'], 
    ['examples/000000581667.jpg']
]

img_x_demo = gr.Interface(
     inference_img_x,     
     gr.Image(type="filepath"), 
     outputs=gr.Image(type="filepath", label="Out"),
    title=title_img_x,
    description=description_img_x,
    article=article_img_x,
    examples=examples_img_x
)

#video SMPL_X
title_vid_x = "HybrIK_vid_X"
description_vid_x = "HybrIK_X demo for gradio"
article_vid_x = "<p style='text-align: center'><a href='https://github.com/Jeff-sjtu/HybrIK/tree/main'>Github Repo</a></p>"
examples_vid_x = [   
['examples/dance.mp4'], 
]

vid_x_demo = gr.Interface(
     inference_vid_x,     
     gr.Video(), 
     outputs=gr.Video(label="Out"),
    title=title_vid_x,
    description=description_vid_x,
    article=article_vid_x,
    examples=examples_vid_x
)



demo = gr.TabbedInterface([img_demo, vid_demo , img_x_demo , vid_x_demo ], ["image demo", "video demo", "image X demo" ,"video X demo"])
demo.launch()
