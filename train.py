from utils import train, save_model
import argparse
   

def main()->int:
    parser = argparse.ArgumentParser(prog="Trainer",
                                     description="My Custom tool for training computer vision models from pretrained models")
    parser.add_argument("data_dir", help="Path to the dataset")
    parser.add_argument("--arch", default='efficientnet_b0',help="The pretrained model you want to use for training")
    parser.add_argument("--hidden_units", type=int,default=256,help="The number of hidden units for the classifier model")
    parser.add_argument("--epochs", type=int,default=20,help="The number of hidden units for the classifier model")
    parser.add_argument("--learning_rate", type=float,default=0.001,help="The pretrained model you want to use for training")
    parser.add_argument("--save_dir",help="The path where your model would be saved after training")
    parser.add_argument("--print_every",type=int,default=2,help="The path the amount")
    parser.add_argument("--gpu",action='store_true',help="Run this option if you want to train on gpu")
    # Parse arguments
    args = parser.parse_args()
    device = 'cpu'
    if args.gpu:
        device ='cuda'
    
        
    model,architecture,hidden_units,optimizer,class_to_idx=train(args.data_dir,args.print_every,hidden_units=args.hidden_units,architecture=args.arch,
          learn_rate=args.learning_rate,epochs=args.epochs,device=device)

    if args.save_dir:
        save_model(args.save_dir,model,architecture,hidden_units,optimizer,class_to_idx)    
   

if __name__ == "__main__":
    main()
    
    