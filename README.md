

## MixRNet(Using mixup as regularization and tuning hyper-parameters for ResNets)
Using mixup data augmentation as reguliraztion and tuning the hyper parameters of ResNet 50 models to achieve **94.57%** test accuracy on CIFAR-10 Dataset. [Link to paper](https://arxiv.org/abs/2111.11616)
| **network** | **error %** | 
|-------------|-------------|
| resnet-50   | 6.97        | 
| resnet-110  | 6.61        |
| resnet-164  | 5.93        |
| resnet-1001 | 7.61        |
| **This method** | **5.43** |


#### Overview

 - Change the wandb api key to valid api key.
 - Python 3.8 and pytorch 1.9 (works on older versions as well)
 - main.py is to train model
 - sweep.py and sweep_config.py are for hyperparameter optimization for experiment tracking wandb is used please change api key
 - pred.py is to run the trained model on the custom data. (Appropriately provide model paths)


##### Important
If you want to run sweep.py then you must use wandb apikey and if you want to run main.py use wandb to log the experiment for comparision else comment out wandb part.

## Training

```

# Start training with:

python main.py (Added --run_name optional argument for better tracking experiments)

  

# You can manually resume the training with:

python main.py --resume --lr=0.01

```

## Hyperparameters sweep

```

# Start sweep with:

python sweep.py

  

# Provide appropriate hyperparameters range in sweep_config.py (Config written in py file to use the power of math package for sweep configs)

```

## Running on custom dataset

```

# Convert traget data of (N*32*32*3) into (N*3*32*32) shape and pass through the model:

python pred.py (Provide path of the saved models)


```

### Other files

- mixup.py contains functions to claculate loss of mixup predictions as you cant use nn.CrossEntropyLoss
- utils.py contain somehelper functions
- dataloader.py is a torch class based dataloader of our train data (CIFAR-10 data)
- private_loader.py is a torch class based dataloader of our private data.
- Transformations are done using torchtransforms in main.py and sweep.py files depending on usage.

## Under-Development

- Addind tensorbaord implementation under new branch
- Hyper-parameters sweep and distributed training using ray.io
