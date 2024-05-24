# make the necessary imports
from utils import load_model,load_categories,predict
import argparse
import torch


def main()->int:     
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
    model,class_to_idx = load_model(args.checkpoint)
    # load the category names
    cat_to_name = load_categories(args.category_names)
    # device        
    if device == 'gpu':
        device = ('gpu' if torch.cuda.is_available else 'cpu')
        
    # run the model
    predict(args.image_path,model,device,class_to_idx,cat_to_name,topk=args.top_k) 

if __name__ == "__main__":
    main()