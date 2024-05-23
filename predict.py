# make the necessary imports
import torch
from PIL import Image
import json
import numpy as np
import argparse

# load the model
def load_model(check_point_path):
    model = torch.load(check_point_path)
    return model 

# load categories
def load_categories(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
# process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    pil_image = pil_image.resize((256,256))
    # set the crop size
    crop_size = 224
    # Get the details of the image
    width, height = pil_image.size
    # calculate the cropping box
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2 
    right = (width + crop_size) / 2
    bottom = ( height + crop_size ) / 2
    # crop the image
    pil_image = pil_image.crop((left,top,right,bottom))
    
    return pil_image


# Prediction
def predict(image_path, model,device, cat_to_name,topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # load the image and resize
    image = process_image(image_path)
    model.to(device)
    # switch to evaluation mode
    model.eval()
    # convert pil image to numpy then to torch tensor and reshape to fit model input shape
    input_image = torch.from_numpy(np.array(image)).view(1,3,224,224)
    input_image = input_image.type(torch.FloatTensor).to(device)
    # pass the image to the model
    with torch.no_grad():
        logps = model.forward(input_image)
    # get the probabilities
    ps = torch.exp(logps)
    # get the top class and probs and return
    top_p,top_class = ps.topk(topk,dim=1)
    # convert the classes to names
    # print(top_class[0])
    top_class_numpy = top_class.cpu().numpy()
    classes = top_class_numpy[0]
    names = [cat_to_name[str(c+1)] for c in classes ]
    top_p_numpy = top_p.cpu().numpy()
    # plot the prediction of the image
    probabilities = top_p_numpy[0]
    for i, name in enumerate(names):
        print(f'{i+1}. prediction: {name} ......  probability: {probabilities[i]} ')
        
        
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog='Prediction', 
                                     description="These program loads a model from a checkpoint and uses it for prediction")
    parser.add_argument("image_path",help='The path to the image you want to use for prediction')
    parser.add_argument("checkpoint",help="The path to the model you want to use for prediction")
    parser.add_argument("--top_k",type=int,default=1,help="How many top predictions you want to print for a sinlge image")
    parser.add_argument("--category_names",default='cat_to_name.json',help='The mapping of index of categories to thier respective names')
    parser.add_argument("--gpu",action='store_true',help="Run this option if you want to run on gpu")

    # Parse arguments
    args = parser.parse_args()
    device = 'cpu'
    if args.gpu:
        device = 'cuda'
        
    # load the model
    model = load_model(args.checkpoint)
    # load the category names
    cat_to_name = load_categories(args.category_names)
    # device        
    if device == 'gpu':
        device = ('gpu' if torch.cuda.is_available else 'cpu')
        
    # run the model
    predict(args.image_path,model,device,cat_to_name,topk=args.top_k)