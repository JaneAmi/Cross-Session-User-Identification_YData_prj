# Cross-Session User Identification project

User identification algorithm employing Siamese Neural Networks and FAISS database, enhancing session-to-session user tracking accuracy

This notebook demonstrates an information retrieval part of the project Cross Session User Identification.

**Project description**

A common problem in e-commerce websites is identifying anonymous users when they return to the website without logging in. But when users visit website the live traces: they usually use same device, browser, choose similar sizes, they have preferred brand and they tend to have similar behavioral patterns across the session. 

The idea of this project is to find hidden information about users that can help to identify returned users. 


**Data**

The original data - is a information about pageviews of visitors of a e-commerce website during one year. Due to NDA this repository hasn't original data.

**Solution**

The final pipeline of the project is presented on the figure 1. The parts of the project, that are presented in this repository are shown in color.

<div align="center">
    <img src="https://github.com/emunaran/xai-compare/raw/main/docs/images/Consistency_wf.png" alt="Consistency Measurement Workflow" width="700"/>
    <p style="color: #808080;">Project Workflow</p>
</div>
FIGURE 1


**Unshown parts of the project**

1. **EDA**
- During the EDA we found many information that helps to identify users, like usually visitors use 1 or 2 devices to visit the website, same statistics with browsers. Users have preferable brand and sizes.  
- Also, we found that data contains external identifiers that appears in about 60% of sessions, we used these identifiers for the heuristic approach. 

2. **Data preprocessing**
 - Grouping by session: each entry in the original data was a pageview, so we grouped it by session. To do that wisely it requires a lot of preprocessing, e.g. summarization, count, choose mode etc. 

 - Hadling categorical values: For most categorical features we choosed about 10 best values that help to distinct users and applied one-hot encoding to them.

 - Normalization: the data was normalized using MinMaxScaler

 3. **Heuristic approach**
 The heurisct approach was realized by creating a dictionary with the external identifiers from the train data. When there is a new session that has at least one of the external identifiers - it is checked within this dictionary. The performance of this approach is 0.60 recall at 1 and 0.63 recall at 3. The difference can be explained by using ip-address as one of the external identifiers. In this case understandably one value of ip-address can appears in more that one user.