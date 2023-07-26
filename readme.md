# BacklitNet_readme.md

## **Introduction**

This is a README for the implementation of the paper "BacklitNet". It includes explanations of how to run the project and what each directory is for.

## **Getting Started**

### **Prerequisites**

Before you proceed, make sure you have the following installed:

- **[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)** (to create virtual environments)
- **[Git](https://git-scm.com/downloads)** (to clone the repository)

### **Creating Conda Environment**

Clone the repository to your local machine OR download whole CODE document from Teams

```bash
git clone https://github.com/leehao921/Backlit.git
```

Change to the project directory:

```bash
cd CODE
```

Create a new Conda environment with Python:

```lua
conda create -n project_env python=3.11.2
```

Activate the environment:

```
conda activate project_env
```

### **Installing Dependencies**

To install the required dependencies from the **`requirements.txt`** file, run the following command:

```
pip install -r requirements.txt
```

# Directory name and Explanation

/full_resInput：

Use this to store the full-resolution input of the image. 

/low_resInput

This is used to store downsampled images, which is compressed data (from 5472*3648 to 384*384)

/groundTruth

Use this to store the full-resolution output of the image. 

## High Resolution Branch

- [ ]  save image to guidance map directory
- [ ]  slice → output and loss func.
- [x]  pixel_wise_network

pixel_wise_network.ipynb:

the pixel wise network flow and testing

rotate_full_resInput.py:

make all image into same 5472*3648 by rotate 90 degree, to ensure it can fit the model 

## Low Resolution Branch

- [x]  downsample
- [x]  feature extractor
- [x]  semantics perception block
- [ ]  lighting acquisition block
- [ ]  fusion adjustment block

Downsample.py:

use to compress img from 5472*3648 to 384*384 and save it 

data_set.py:

The Dataset class is used to preprocess (transform) data before it enters the CNN model.

FlowTesting.ipynb:

FlowTesting encompasses the steps of importing data, preprocessing it, passing it through the model, and finally displaying the results for testing. 

Extractor: 

use Feature_Extractor model to run:

 Semantics:

use Semantics_perception+Feature_Extractor model to run 

Lighting:

use Lighting Acquisition + Semantics perception+Feature_Extractor model to run 

Fusion:

use Fusion adjustment + Lighting Acquisition + Semantics perception+Feature Extractor model to run 

> Note that the original architecture may have some error
>
