import torch
from PIL import Image
from imutils import paths


# Initializes the model at the start, run model = load_model
def load_model():
    model = torch.hub.load('./', 'custom', path='yolov5s_best.pt', source='local')
    return model

# Takes in the image from the 'test' folder, and outputs the predicted label - sample at the end
# Images with predicted bounding boxes are saved in the runs folder
def predict_image(image,model):
    img = Image.open('test/' + image)
    results = model(img)
    results.save('runs')
    pred_list = results.pandas().xyxy[0]['name'].to_numpy()
    pred = 'NA'
    for i in pred_list:
        if i != 'Bullseye':
            pred = i
    return pred

# Stitches the previously predicted images in the folder together and saves it into runs/stitched folder
# This function can be called by itself
def stitch_image():
    imgFolder = 'runs'
    newPath = 'runs/stitched.jpg'
    imgPath = list(paths.list_images(imgFolder))
    images = [Image.open(x) for x in imgPath]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        stitchedImg.paste(im, (x_offset,0))
        x_offset += im.size[0]
    stitchedImg.save(newPath)

## Load
# model = load_model()

## Predict
# image = 'filename.jpg'
# print(predict_image(image, model))



