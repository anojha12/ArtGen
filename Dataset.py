import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import kagglehub

# Download latest version
path = kagglehub.dataset_download("superpotato9/dalle-recognition-dataset")

print("Path to dataset files:", path)


# class Dataset:
#     def __init__(self, root):
#         self.root = root
#         self.images = []
#         self.labels = []
#         self.load_dataset()

#     def load_dataset(self):