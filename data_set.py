import os
from torchvision.transforms import ToTensor
from PIL import Image
class Dataset():
    def __init__(self, data_path, transform=None):
        # self.input_path = os.path.join(data_path, "input")  # Path to the input data directory
        # self.output_path = os.path.join(data_path, "output")  # Path to the output data directory
        print(data_path)
        print(os.listdir(data_path))
        self.file_list = os.listdir(data_path)
        self.transform = transform
        self.filepath = data_path
        print(data_path)
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input_file_name = self.file_list[idx]
        output_file_name = input_file_name  
        # input and output file names are the same
        input_file_path = os.path.join(self.filepath, input_file_name)
        output_file_path = os.path.join(self.filepath, output_file_name)
        
        
        transform = ToTensor()
        # Load and preprocess the input data
        image = Image.open(input_file_path)
        inputs = transform(image)

        # Load and preprocess the output data
        image = Image.open(output_file_path)
        targets = transform(image)


        return inputs, targets
    
