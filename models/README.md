We leveraged the Siamese Neural Network (SNN) approach to generate embeddings.


Below is the architecture that yielded the best results, but many others combinations were tested and you can find all description here.

<div align="center">
    <img src="../docs/images/Final_SNN_architecture.png" alt="Best result SNN architecture" width="700"/>
    <p style="color: #808080;">Best result SNN architecture</p>
</div>


All classes for models, datasets and functions for training are presented [here](siamese_nn_utils.py)

<details><summary><b><big>Used Siamese Neural Network (SNN) architechtures</big></b></summary>

While training the model we tried SNN for pairs and for triplets. 

1. SNN for pairs:

<div align="center">
    <img src="../docs/images/SNN_pairs.png" alt="SNN pairs" width="700"/>
    <p style="color: #808080;">SNN architecture for pairs</p>
</div>

Used losses for training:
**Contrastive loss based on Euclidean distance**:

*L =  y * D² + (1 - y) * max(0, m — D)²*


*D* = Euclidean distance

**Contrastive loss based on Cosine Similarity**

*L = y * CD² + (1 - y) * max(0, m — CD)²*

*CD = 1 - cosine_similarity*


*m* - margin
*y = 1* for similar pairs
*y = 0* for dissimilar pairs

2. SNN for triplets:

<div align="center">
    <img src="../docs/images/SNN_triplets.png" alt="SNN triplets" width="700"/>
    <p style="color: #808080;">SNN architecture for triplets</p>
</div>

**Triplet Margin Loss**:
*L(a, p, n) = max{d(ai, pi) - d(ai, ni) + m, 0}*

*a* - anchor session
*p* - positive example - a session from the similar user
*n* - negative example - a session from the different user

*m* - margin
*d* - Euclidean distance
</details>

<details><summary><b><big>Used model architecturess</big></b></summary>

1. TabNet

2. 3 fully connected layers with ReLU

<div align="center">
    <img src="../docs/images/3_layers_with_relu.png" alt="3 fc with relu" width="400"/>
    <p style="color: #808080;">3 fully connected layers with ReLU</p>
</div>


3. 3 fully connected layers with Sigmoid

4. 4 fully connected layers
</details>


<details><summary><b><big>Data Division Approaches
</big></b></summary>

We employed different strategies for splitting our data into training, testing, and validation sets.

1st

<div align="center">
    <img src="../docs/images/test_train_val_1.png" alt="Data division 1st" width="700"/>
    <p style="color: #808080;">Data division by users</p>
</div>


2nd

<div align="center">
    <img src="../docs/images/test_train_val_2.png" alt="Data division 1st" width="700"/>
    <p style="color: #808080;">Train, test and validation sets contain sessions from the same users</p>
</div>
</details>
