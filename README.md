# PreTKcat
Feel free to contact me via email at mr.vincentcai@gmail.com if you encounter any issues or have any questions.

## Introduction of PreTKcat.
The enzyme turnover number ($k_{cat}$) is crucial for understanding enzyme kinetics and optimizing biotechnological processes. However, experimentally measured $k_{cat}$ values are limited due to the high cost and labor intensity of wet-lab measurements, necessitating robust computational methods. To address this issue, we propose PreTKcat, a framework that integrates pre-trained representation learning and machine learning to predict $k_{cat}$ values. PreTKcat utilizes the ProtT5 protein language model to encode enzyme sequences and the MolGNet molecular representation learning model to encode substrate molecular graphs. By integrating these representations, the ExtraTrees model is employed to predict $k_{cat}$ values. Additionally, PreTKcat accounts for the impact of temperature on $k_{cat}$ prediction. In addition, PreTKcat can also be used to predict enzyme-substrate affinity, i.e. km values. Comparative assessments with various state-of-the-art models highlight the superior performance of PreTKcat. PreTKcat serves as an effective tool for investigating enzyme kinetics, offering new perspectives for enzyme engineering and its industrial uses.

## Here is the framework of PreTKcat.
![Figure_1](https://github.com/user-attachments/assets/6c9bc5c8-82a9-45cc-b05d-66117953dde0)

## Usage
1.The dataset is in datasets/DLTKcat_data

2.Download pre_trained model
The molecular graph representation learning model MolGNet can be downloaded [here] (https://github.com/pyli0628/MPG)
Other pre-training models mentioned in the paper can be downloaded from [huggingface] (https://huggingface.co/)

3.Calculate the Enzyme and Substrate representation./Train and test the PreTKcat
```
python PreTKcat.py
```
