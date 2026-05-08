# IIVP Challenge - Group 7

Welcome to our solution to the challenge of recognizing Hindi numbers.  

## Data Setup (Important)

The dataset is not included in this repository because there are too many files. Before running the code place the dataset files in the `data/` directory at the project root. 

The structure should look like this:
```
data/
    train.csv
    test.csv
    manual label.csv
    sample_submission.csv
    train/
        train/
            0/
            1/
            ...
    test/
        test/
```

## How to Run the Code

The main script to test our code is `run_test.py`. It uses pre-trained ensemble weights to generate predictions on the test images.

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the test script - `run_test.py`:
   
   This will load the pre-trained weights - `main_triple_stacking_longer_swa.pth` , process the images in the `data/test/` directory, and output the results to `final_submission.csv`.

## Training from Scratch

To train the model from scratch. You need to do the following steps. 

1. Prepare the cross-validation splits. This only needs to be run once - `setup_folds.py`:
2. Start the training process - `main.py`:
   

## Repository Structure

Below is an overview of the codebase:

- `run_test.py`: The main testing script. Loads weights and predicts test labels.
- `main.py`: The main training loop. Trains the models and exports the `.pth` weight files.
- `setup_folds.py`: Service script to divide the training dataset into folds.
- `data_analysis.py`: Initial script for data exploration and sanity checks.
- `src/models/factory.py`: Factory design pattern class to easily integrate different model architectures.
- `src/dataset/trainer.py`: Contains training and validation loops used in our models.
- `src/models/` Ensemble logic and models.Implementation of different ensemble approaches and different models we tried.
- `experiments/`: Contains Jupyter notebooks explaining our methodology, benchmark comparisons, and initial baselines.
