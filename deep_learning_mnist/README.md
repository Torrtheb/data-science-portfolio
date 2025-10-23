# MNIST MLP – Deep Learning Fundamentals Project

This project trains a configurable multi‑layer perceptron (MLP) on MNIST to practice PyTorch and PyTorch Lightning, run ablations, and evaluate a final model. It includes a clean training pipeline, utilities for evaluation and visualization, and optional hyperparameter search with Optuna and data augmentation.

## Objectives
- Build an MLP that performs well on MNIST.
- Practice PyTorch/Lightning, loaders, callbacks, and logging.
- Run ablations and an Optuna search to choose hyperparameters.
- Evaluate on a held‑out test set and generate reports/plots.

## Environment
- Python 3.10+ (tested on 3.13)
- PyTorch, PyTorch Lightning, torchvision, numpy, pandas, scikit‑learn, seaborn, matplotlib, optuna (optional).

GPU support:
- NVIDIA CUDA (Linux/Windows) or Apple Silicon MPS (macOS) if available. The code auto‑selects gpu → mps → cpu. For dataloaders in notebooks, num_workers is generally set to 0 to try to avoid multiprocessing crashes. 

Reproducibility: 
- Seeds: the training pipeline sets seeds where applicable. To ensure reproducibility, it is important to pass a fixed seed where exposed.

## Data
The MNIST digit recognizer dataet is found on Kaggle: https://www.kaggle.com/competitions/digit-recognizer/overview.

## Notebook structure: 
- Imports, seeding and GPU access verification
- Kaggle dataset download
- Initial dataset exploration
- Test, train, validation split
- Data processing
- Logging setup
- Baseline Softmax Model
- Baseline Multi Layer Perceptron Model
- Experiments
- Optuna hyperparameter tuning
- Results evaluation
- Evaluation on test data
- Saving best model
- Conclusion

Functions are found in mnist.py, and data is found in the data/ folder. 

## Key results
In this project, a reproducible MLP (multilayer perceptron model) has been built using train, validation and test tensors. It was found that performing slight image augmentations (10 degree rotation, 5% image shifting, slanting an image by 5 degrees, zooming the image by 5%). 

Single factor ablation experiments were performed to examine individual parameter contributions to overall validation accuracy. It was found that a learning rate of 0.003 had the most impact on validation accuracy (98.4%, base: 98.2%). Optuna was used to tune these hyperparameters together, and found the following parameter combination to be optimal (validation accuracy of 98.6%): 
- stochastic gradient descent optimizer, 
- learning rate of 0.039, 
- batch size of 64
- GELU activation function,
- Dropout rate of 0, 
- cross entropy loss function
- hidden layer architecture of (512,256,128), 
- focal loss function.

On the test data, the best local validation accuracy was found to be 0.988, which corresponds to a Kaggle score of 0.988. Most common digit confusions were with the number 9, for example: true label 9 predicted as a 7. 


## Suggested Improvements
- Adding CNN baselines for comparison, 
- Adding other image augmentations such as perspective, light blur, random erasing, 
- Increasing training time and patience, 
- Experiment tracking with weights and biases. 

# Acknowledgements
- https://www.kaggle.com/competitions/digit-recognizer/overview