import os
import re
import pandas as pd

class KittiDataset():
    def __init__(self, path):
        self.path = path
    
    def load_train_val(self):
        trainPath = self.path + "data_depth_annotated/train"
        valPath = self.path + "data_depth_annotated/val"

        dataTrain = []
        dataVal = []

        for root, _, files in os.walk(trainPath):
            if "image_02" in root:
                for file in files:
                    if "depth" in file:
                        filePath = re.findall(r"(\d{4}\_\d{2}\_\d{2}\_drive\_\d{4}\_sync)", os.path.join(root, file))[0]
                        fileName = file
                        dataTrain += [{
                            "image" : self.path + "raw_data/" + filePath[:10] + "/" + filePath + "/image_02/data/" + fileName.replace("_depth.png", ".png"),
                            "depth_map" : os.path.join(root, fileName)
                        }]
        
        for root, _, files in os.walk(valPath):
            if "image_02" in root:
                for file in files:
                    if "depth" in file:
                        filePath = re.findall(r"(\d{4}\_\d{2}\_\d{2}\_drive\_\d{4}\_sync)", os.path.join(root, file))[0]
                        fileName = file
                        dataVal += [{
                            "image" : self.path + "raw_data/" + filePath[:10] + "/" + filePath + "/image_02/data/" + fileName.replace("_depth.png", ".png"),
                            "depth_map" : os.path.join(root, fileName)
                        }]
        
        return pd.DataFrame(dataTrain), pd.DataFrame(dataVal)

class KittiDatasetGenerate(KittiDataset):
    def __init__(self, path):
        super().__init__(path)
        
    def load_train_val(self):
        trainPath = self.path + "data_depth_annotated/train"
        valPath = self.path + "data_depth_annotated/val"

        dataTrain = []
        dataVal = []

        for root, _, files in os.walk(trainPath):
            if "image_02" in root:
                for file in files:
                    if "depth" not in file:
                        filePath = re.findall(r"(\d{4}\_\d{2}\_\d{2}\_drive\_\d{4}\_sync)", os.path.join(root, file))[0]
                        fileName = file
                        dataTrain += [{
                            "image" : self.path + "raw_data/" + filePath[:10] + "/" + filePath + "/image_02/data/" + fileName,
                            "depth_map" : os.path.join(root, fileName)
                        }]
        
        for root, _, files in os.walk(valPath):
            if "image_02" in root:
                for file in files:
                    if "depth" not in file:
                        filePath = re.findall(r"(\d{4}\_\d{2}\_\d{2}\_drive\_\d{4}\_sync)", os.path.join(root, file))[0]
                        fileName = file
                        dataVal += [{
                            "image" : self.path + "raw_data/" + filePath[:10] + "/" + filePath + "/image_02/data/" + fileName,
                            "depth_map" : os.path.join(root, fileName)
                        }]
        
        return pd.DataFrame(dataTrain), pd.DataFrame(dataVal)