## Requirements
* [TensorFlow 2.2.0+](https://www.tensorflow.org/)
* [Python 3.6+](https://www.continuum.io/downloads)
* [Scikit-learn 0.23.2+](https://scikit-learn.org/stable/)
* [Nibabel 3.0.1+](https://nipy.org/nibabel/)

## Downloading datasets
To download Alzheimer's disease neuroimaging initiative dataset
* https://adni.loni.usc.edu/

To download Gwangju Alzheimer’s and Related Dementia

## How to Run
### Counterfactual Map Generation
Mode: #0 Learn, #1 Explain

1. Learn: pre-training the predictive model
>- `CMG_config.py --mode="Learn"`
>- Set the mode as a "Learn" to train the predictive model

2. Explain: Counterfactual map generation using a pre-trained diagnostic model
>- `CMG_config.py --mode="Explain" --dataset=None --scenario=None`
>- Change the mode from "Learn" to "Explain" on Config.py
>- Set the classifier and encoder weight for training (freeze)
>- Set the variables of dataset and scenario for training

### AD-Effect Map and LiCoL
1. AD-effect map acquisition based on manipulated real-/counterfactual-labeled gray matter density maps
>- `AD-effect Map Acquisition.ipynb`
>- This step for the AD-effect map acquisition was implemented by using the Jupyter notebook
>- Execute markdown cells written in jupyter notebook in order

2. LiCoL
>- `LiCoL_ALL.py --datatset=None --scenario=None --data_path==None`
>- Set the variables of dataset and scenario for training
>- For example, dataset="ADNI" and scenario="CN_AD"
>- Modify the data path for uploading the dataset (=line 234)
