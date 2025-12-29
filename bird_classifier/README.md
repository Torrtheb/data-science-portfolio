# Fine-Grained Bird Species Classification with Transfer Learning and Few-Shot Learning


## Objectives
- Perform EDA on the NABirds image dataset to explore bird species images, identify data quality issues, and design appropriate preprocessing and augmentation strategies.
- Build a bird species classifier using few-shot learning techniques (simulating a scenario with limited data) with iterative pseudo-labeling to expand the training set from limited labeled samples.
- Fine-tune a deep learning classifier using transfer learning on the entire training dataset (555 North American bird species).

## Environment
- Python 3.13
- PyTorch, torchvision, NumPy, pandas, scikit-learn, scikit-image, seaborn, matplotlib, Optuna, LIME, Albumentations, DeepLake
- GPU support: Training performed on Google Colab GPU with local development on Apple Silicon MPS with fallback to CPU.
- Reproducibility: The training pipeline sets seeds where applicable to ensure reproducibility. 

## Data 
This dataset is sourced from DeepLake: https://app.activeloop.ai/activeloop/nabirds-dataset-train and consists of approximately 48,000 photographs of 555 North American bird species. Images vary in size (most in landscape mode), contain 3 RGB channels, and include bounding box annotations to localize birds in each image. The training set is relatively balanced and contains 23,928 samples and the validation set contains 24,633 samples.


## Project Structure

Note: to see notebooks with outputs, download files from the notebooks_with_outputs folder. 

- **EDA** (eda.ipynb, python_files/eda.py):
  - Imports and setup
  - Image corruption verification
  - Duplicate image examination
  - Image channel and size examination
  - Image class distribution analysis
  - Image visualization with bounding boxes
  - Image quality analysis:
    - Brightness and contrast
    - Blur and sharpness (variance of the Laplacian)
    - Color distribution
    - Background complexity (edge intensity)
    - Texture (entropy and local standard deviation)
    - Correlations between metrics
  - Inter-class similarity analysis (UMAP visualization)
  - Image transformation experiments (CLAHE, gamma correction, sharpening, bbox cropping)
  - Augmentation pipeline design
  - EDA conclusions
  * Note: files to remove duplicates for later use are created in this notebook (clean_indices.npz)

- **Few-Shot Learning** (fewshot_final.ipynb, python_files/fewshot.py, utilities_fewshot.py):
  - Setup and configuration
  - Loading and splitting training data (max 5 images per class)
  - Multi-backbone feature extraction (EfficientNet-B4, ResNet-50, ViT-B-16)
  - Initial few-shot baseline evaluation
  - Learned projection head (PrototypicalNetwork)
  - Iterative pseudo-labeling to expand training data
  - Model evaluation on holdout set
  - Conclusions

- **Model Fine-Tuning** (model_tune.ipynb, python_files/model_tune.py):
  - Setup and configuration
  - Initial processing visualization
  - Training preparation (data splits, transforms)
  - Classifier head tuning (frozen backbone)
  - Fine-tuning the classifier backbone
  - Hyperparameter tuning with Optuna:
    - Augmentation search (geometric and color transforms)
    - Model hyperparameter search (optimizer, learning rate, regularization, loss function, scheduler)
  - Final training on full dataset
  - Holdout dataset evaluation
  - LIME explainability analysis
  - Conclusions

## Key Results

### Few-Shot Learning Approach

Starting with a maximum of 5 samples per class (about 3,000 total samples), a prototype-based classifier was built using embeddings from EfficientNet-B4, ResNet-50, and ViT-B-16. The best configuration used:
- EfficientNet-B4 backbone with bounding box preprocessing, 
- PrototypicalNetwork with 512 learned embeddings, and
- Iterative pseudo-labeling to expand training data to about 13,000 samples

**Few-Shot Results:**
| Metric | Score |
|--------|-------|
| Accuracy | 72% |
| Precision | 70% |
| Recall | 69% |
| F1 Score | 68% |

### Transfer Learning Approach

After few-shot experiments, the full labeled dataset was used to fine-tune an EfficientNet-B4 classifier:
- **Preprocessing**: Bounding box cropping (best initial method, 73% test accuracy after head tuning)
- **Backbone fine-tuning**: Increased test accuracy to 81%
- **Augmentations** (Optuna-tuned): Horizontal flips, rotations, slight scale changes, noise, cutout (~10% probability). Minimal color augmentation to preserve species-distinguishing color features.
- **Model hyperparameters** (Optuna-tuned): AdamW optimizer, dropout (0.12), cross-entropy loss, cosine learning rate scheduler

**Fine-Tuned Model Results:**
| Metric | Score |
|--------|-------|
| Accuracy | 77% |
| Precision | 76% |
| Recall | 75% |
| F1 Score | 75% |


### Key Findings

- **Inter-class similarity challenges**: Subspecies and visually similar species (e.g., Chickadee variations, Hummingbird species, Dark-eyed Junco subspecies) are frequently confused by both approaches.
- **Color preservation is essential**: Strong color augmentations hurt performance because minute color differences distinguish bird species.
- **Bounding box cropping**: Significantly improves classification by removing background noise and focusing on bird morphology.
- As compared to the few shot learning approach, the fine tuning approach takes longer, but showed a 5% increase in accuracy with higher recall (75%), F1 score (75%), and precision (76%). 

## Improvements

- **Hierarchical classification**: Implementing a two-stage classification (general species → subspecies) could reduce subspecies confusion.
- **Attention mechanisms**: Adding attention layers to help the model focus on discriminative bird regions could improve accuracy.
- **Extended training**: The final model showed continued improvement through epoch 19 of 20, suggesting more training epochs could yield further gains. In addition, fine tuning the model could be more robust with more training time. For example, using a larger sample for hyperparameter search with more head and fine-tuning epochs per iteration could increase metrics and decrease model overfitting. 
- **Robust pseudo-labeling**: Combining confidence scores with prototype distance measures could improve the iterative labeling process.
- **Data augmentation refinement**: Exploring more sophisticated augmentations that preserve species-critical features while increasing robustness.

## Acknowledgements
- Dataset: NABirds on DeepLake (https://app.activeloop.ai/activeloop/nabirds-dataset-train)
- Open source libraries: PyTorch, torchvision, Albumentations, Optuna, LIME, scikit-learn, DeepLake
- ChatGPT and GitHub Copilot for code assistance