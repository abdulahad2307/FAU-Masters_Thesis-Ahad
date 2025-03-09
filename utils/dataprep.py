import os
import csv
import shutil
import pandas as pd

BaseDataPath = r"/home/woody/iwi5/iwi5280h/dataset/data/images"
LabelPaths = {
    "train": r"/home/woody/iwi5/iwi5280h/dataset/data/labels/train.txt",
    "val": r"/home/woody/iwi5/iwi5280h/dataset/data/labels/val.txt",
    "test": r"/home/woody/iwi5/iwi5280h/dataset/data/labels/test.txt"
}

CSVPaths = {
    "train": "/home/woody/iwi5/iwi5280h/dataset/data/train.csv",
    "val": "/home/woody/iwi5/iwi5280h/dataset/data/val.csv",
    "test": "/home/woody/iwi5/iwi5280h/dataset/data/test.csv"
}

## Mapping class numbers to names
class_mapping = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific report",
    6: "scientific publication",
    7: "specification",
    8: "file folder",
    9: "news article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo"
}

# Function to convert label txt to CSV
def convert_txt_to_csv(label_path, csv_path):
    with open(label_path, 'r') as infile, open(csv_path, 'w') as outfile:
        stripped = (line.strip() for line in infile)
        lines = (line.split(",") for line in stripped if line)
        writer = csv.writer(outfile)
        writer.writerows(lines)

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['image'])
    df['image'] = df['image'].astype(str)

    ## Spliting into 'image' and 'class' columns
    df[['image', 'class']] = df['image'].str.split(' ', n=1, expand=True)
    
    ## Converting class column to integer
    df['class'] = df['class'].astype(int)

    return df


for split in ["train", "val", "test"]:
    convert_txt_to_csv(LabelPaths[split], CSVPaths[split])


df_train = load_dataframe(CSVPaths["train"])
df_val = load_dataframe(CSVPaths["val"])
df_test = load_dataframe(CSVPaths["test"])

# Display first few rows of each DataFrame
print("Train Data:\n", df_train.head())
print("Validation Data:\n", df_val.head())
print("Test Data:\n", df_test.head())