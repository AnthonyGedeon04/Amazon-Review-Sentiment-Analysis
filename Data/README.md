# Data

The CSV files are not in this repository. The reviews come from the
[Amazon Echo Dot 2 Reviews Dataset](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-echo-dot-2-reviews-dataset)
on Kaggle (published by PromptCloud), so download them from there.

1. Download the dataset from the Kaggle page (a free Kaggle account is needed).
2. Unzip it and put `Amazon Echo Dot 2 Reviews.csv` in this `Data/` folder.
3. Run `Notebooks/01_data_prep_and_baseline.ipynb`. It creates `train.csv` and `test.csv` here
   (the same fixed split every notebook uses, seed 42).
