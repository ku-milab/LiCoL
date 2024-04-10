# LiCoL
This repository provides a TensorFlow implementation of the following paper:
> **A Quantitatively Interpretable Model for Alzheimer’s Disease Prediction using Deep Counterfactuals**<br>
> [Kwanseok Oh](https://scholar.google.co.kr/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>1</sup>, [Da-Woon Heo](https://scholar.google.co.kr/citations?user=WapMdZ8AAAAJ&hl=ko&oi=ao)<sup>1</sup>, [Ahmad Wisnu Mulyadi](https://scholar.google.co.kr/citations?user=u50w0cUAAAAJ&hl=ko)<sup>2</sup>, [Wonsik Jung](https://scholar.google.co.kr/citations?user=W4y-TAcAAAAJ&hl=ko)<sup>2</sup>, Eunsong Kang<sup>2</sup>, [Kunho Lee](https://scholar.google.co.kr/citations?user=AoXfBv8AAAAJ&hl=ko)<sup>3, 4</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1, 2</sup><br/>
> (<sup>1</sup>Department of Artificial Intelligence, Korea University) <br/>
> (<sup>2</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> (<sup>3</sup>Department of Biomedical Science and Gwangju Alzheimer’s & Related Dementia Cohort Research Center, Chosun University) <br/>
> (<sup>4</sup>Korea Brain Research Institute) <br/>
> Official Version: https://arxiv.org/pdf/2310.03457.pdf <br/>
> 
> **Abstract:** *Deep learning (DL) for predicting Alzheimer’s disease (AD) has provided timely intervention in disease progression yet still demands attentive interpretability to explain how their DL models make definitive decisions. Recently, counterfactual reasoning has gained increasing attention in medical research because of its ability to provide a refined visual explanatory map. However, such visual explanatory maps based on visual inspection alone are insufficient unless we intuitively demonstrate their medical or neuroscientific validity via quantitative features. In this study, we synthesize the counterfactual-labeled structural MRIs using our proposed framework and transform it into a gray matter density map to measure its volumetric changes over the parcellated region of interest (ROI). We also devised a lightweight linear classifier to boost the effectiveness of constructed ROIs, promoted quantitative interpretation, and achieved comparable predictive performance to DL methods. Throughout this, our framework produces an “AD-relatedness index” for each ROI and offers an intuitive understanding of brain status for an individual patient and across patient groups with respect to AD progression.*

## Overall framework
- We propose a novel methodology to develop fundamental scientific insights from a counterfactual reasoning-based explainable learning method. We demonstrate that our proposed method can be interpreted intuitively from the clinician’s perspective by converting counterfactual-guided deep features to the quantitative volumetric feature domain rather than directly inspecting DL-based visual attributions.
- We achieved similar or better performance than DL-based models by designing a shallow network of lightweight counterfactual-guided attentive feature representation and a linear classifier (LiCoL) with the AD-effect ROIs considered to be the distinctive AD-related landmarks via counterfactual-guided deep features.
- By exploiting our proposed LiCoL, we provide a numerically interpretable AD-relatedness index for each patient as well as patient groups with respect to anatomical variations caused by AD progression.

<p align="center"><img width="90%" src="https://github.com/ku-milab/LiCoL/assets/57162425/17ef8f9e-d315-4b13-b3b8-eed65f1f2ecd" /></p>

## Qualitative Analyses
### Illustration of inferred AD-effect and statistical maps in binary and multi-class scenarios
<p align="center"><img width="100%" src="https://github.com/ku-milab/LiCoL/assets/57162425/d9fae7e4-a506-4fdd-b36f-19f7bb29b54f" /></p>
<p align="center"><img width="65%" src="https://github.com/ku-milab/LiCoL/assets/57162425/532c56a6-d412-4822-b89b-1e3d8b6b3c0a" /></p>

### Visualization of a normalized AD-relatedness index over the group-wise (first column) and individuals (second and third columns) 
<p align="center"><img width="100%" src="https://github.com/ku-milab/LiCoL/assets/57162425/2168d64a-6c21-46bc-a6c6-2deabd9ae2be" /></p>

## Quantitative Evaluations
<p align="center"><img width="100%" src="https://github.com/ku-milab/LiCoL/assets/57162425/7438832c-9fff-48ea-a46c-ba6306e8c7e4" /></p>
<p align="center"><img width="50%" src="https://github.com/ku-milab/LiCoL/assets/57162425/44d146ef-5884-4f98-9f8a-6f50ee7ef060" /></p>

## Requirements
* [TensorFlow 2.2.0+](https://www.tensorflow.org/)
* [Python 3.6+](https://www.continuum.io/downloads)
* [Scikit-learn 0.23.2+](https://scikit-learn.org/stable/)
* [Nibabel 3.0.1+](https://nipy.org/nibabel/)

## Downloading datasets
To download Alzheimer's disease neuroimaging initiative dataset
* https://adni.loni.usc.edu/

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

## Citation
If you find this work useful for your research, please cite the following paper:

```
@article{oh2023quantitatively,
  title={A Quantitatively Interpretable Model for Alzheimer's Disease Prediction Using Deep Counterfactuals},
  author={Oh, Kwanseok and Heo, Da-Woon and Mulyadi, Ahmad Wisnu and Jung, Wonsik and Kang, Eunsong and Lee, Kun Ho and Suk, Heung-Il},
  journal={arXiv preprint arXiv:2310.03457},
  year={2023}
}
```

## Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) No. 20220-00959 ((Part 2) Few-Shot Learning of Causal Inference in Vision and Language for Decision Making) and No. 20190-00079 (Department of Artificial Intelligence (Korea University)). This study was further supported by KBRI basic research program through Korea Brain Research Institute funded by the Ministry of Science and ICT (22-BR-03-05).
