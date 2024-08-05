# Cross-Session User Identification project

This repository demonstrates the information retrieval component of the Cross Session User Identification project, which utilizes Siamese Neural Networks and a FAISS database to enhance session-to-session user tracking accuracy.

## Project overview

Identifying anonymous users who return to e-commerce sites without logging in is a common challenge. These users often leave identifiable traces such as device usage, browser preferences, and shopping behaviors, which can be leveraged to recognize return visits.


## Data

The project uses data detailing visitor pageviews on an e-commerce site over one year. Due to confidentiality agreements, the original data is not included in this repository.

## Solution

The complete project workflow is visualized below. The components included in this repository are highlighted.

<div align="center">
    <img src="docs/images/Process scheme.png" alt="Consistency Measurement Workflow" width="700"/>
    <p style="color: #808080;">Project Workflow</p>
</div>


## Unshown Parts of the Project

1. **Exploratory Data Analysis (EDA)**
- EDA revealed many information that helps to identify users, like usually visitors use 1 or 2 devices to visit the website, same statistics with browsers, they have preferable brand and sizes, etc.
- Approximately 60% of sessions included external identifiers, guiding the heuristic approach. 

2. **Data preprocessing**

 - *Grouping by session*: each entry in the original data was a pageview, so we grouped it by session. To do that wisely it requires a lot of preprocessing, e.g. summarization, count, choose mode etc. 

- *Handling Categorical Values*: For most categorical features, we identified the top 10 values that best distinguish users and applied one-hot encoding to these selected categories.

 - *Normalization*: the data was normalized using MinMaxScaler

The functions created for the preprocessing are presented [here](data_pipeline)


 3. **Heuristic approach**
The heuristic approach involves creating a dictionary from the training data using external identifiers. For each new session containing any of these identifiers, it’s cross-referenced against this dictionary. The approach achieves a recall of 0.60 at 1 and 0.63 at 3. The variation in recall is attributed to using IP addresses as identifiers, which can be linked to multiple users, thereby affecting the recall rate.

The functions created for the heuristic approach are presented [here](models/heuristic_utils.py)

 ## Model training

The Siamese Neural Network approach was utilized to train the model. To achieve decent results, various combinations of models and hyperparameters were tested. All experiments are described and presented [here](models).

## Vector Database

The functions created for utilising FAISS vector database are presented [here](evaluation/faiss_umap_utils.py)

 ## Conclusion

The applied information retrieval approach appears very promising. To achieve better results, the following steps should be taken:

- Extract more meaningful features from the original data.
- Create new features representing consequences.
- Enhance the handling of categorical features.
- Increase the model training time.


 ## References

 - [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442)
 - [How To Train Your Siamese Neural Network](https://towardsdatascience.com/how-to-train-your-siamese-neural-network-4c6da3259463)