import numpy as np

import torch
from tqdm import tqdm

import faiss, umap

import matplotlib.pyplot as plt



def create_faiss_db(data, model=None, tabnet=False):
    """
    Creates a FAISS database from the given data using optional model embedding.

    Args:
    data (pd.DataFrame): DataFrame containing the user data and features.
    model (torch.nn.Module, optional): Model to generate embeddings.
    tabnet (bool): Flag to indicate if the model is a TabNet model.

    Returns:
    tuple: A tuple containing the FAISS index and list of user IDs.
    """
    if model:
        # Convert DataFrame to numpy and ensure type is float32 for PyTorch
        feature_data = data.iloc[:, 2:].to_numpy().astype('float32')
        input_tensor = torch.tensor(feature_data, dtype=torch.float32)

        # Handle batch processing for TabNet and other models differently if required
        if tabnet:
            # Assuming TabNet returns a batch of embeddings directly
            embedded_data = model(input_tensor)[0].detach().numpy()
        else:
            # General model that processes single sample at a time
            embedded_data = np.vstack([model(torch.tensor(x.reshape(1, -1), dtype=torch.float32)).detach().numpy()
                                       for x in feature_data])

    else:
        embedded_data = data.iloc[:, 2:].to_numpy().astype('float32')

    user_ids = data['user_id'].tolist()

    # Normalize vectors for cosine similarity
    embedded_data = np.ascontiguousarray(embedded_data)
    faiss.normalize_L2(embedded_data)

    # Create FAISS index for cosine similarity (inner product on normalized data)
    vector_dimension = embedded_data.shape[1]
    index = faiss.IndexFlatIP(vector_dimension)  # Use IndexFlatIP for inner product (cosine similarity)
    index.add(embedded_data)

    return index, user_ids



def evaluate_faiss_db(k, test_set, faiss_index, user_ids, model=None, tabnet=False, verbose=True):
    
    answ = []
    answ_dict = {'Distance': [], 'Result': [], 'K': []}

    
    for i in tqdm(range(len(test_set))):

        if model:

            if tabnet:
                input_tensor = torch.tensor(test_set.iloc[:, 2:].to_numpy(), dtype=torch.float32)
                embedded_data_test = model(input_tensor)[0].detach().numpy() #.squeeze(0)

            else:
                input_tensor = torch.tensor(test_set.iloc[i, 2:].to_numpy(), dtype=torch.float32).unsqueeze(0)

                embedded_data_test = model(input_tensor).detach().squeeze(0).numpy()
                
            if embedded_data_test.ndim == 1:
                embedded_data_test = embedded_data_test.reshape(1, -1)

        else:
            embedded_data_test = test_set.iloc[i, 2:].values.reshape(1, -1)

        distances, indices = faiss_index.search(embedded_data_test, k)

        # print("Original user_id:", test_set_db['user_id'].iloc[i])
        # print("Nearest sessions:")
        y_user_ids = []

        for ni in range(k):
            idx = indices[0][ni]
            # print(f"Index in FAISS: {idx}, User ID: {user_ids[idx]}, Distance: {distances[0][ni]}")
            y_user_ids.append(user_ids[idx])
            answ_dict['Distance'].append(distances[0][ni])
            answ_dict['Result'].append(test_set['user_id'].iloc[i] == user_ids[idx])
        
            answ_dict['K'].append(k)

        res = test_set['user_id'].iloc[i] in y_user_ids
        
        # print(res)
        answ.append(res)
        # print()   

    if verbose:
        print(sum(answ) / len(answ))
    
    return answ_dict




def umap_plot(data, umap_hparams, model=False, save_to_file=False, f_path=None, tabnet=False):
    
    label = data['user_id']

    if model:
    
        if tabnet:
            input_tensor = torch.tensor(data.iloc[:, 2:].to_numpy(), dtype=torch.float32)
            embedding = model(input_tensor)[0].detach().squeeze(0).numpy()

        else:
            input_tensor = torch.tensor(data.iloc[:, 2:].to_numpy(), dtype=torch.float32).unsqueeze(0)
            embedding = model(input_tensor).detach().squeeze(0).numpy()

    else:
        embedding = data.iloc[:, 2:].to_numpy()


    # Create and fit UMAP model
    umap_model = umap.UMAP(**umap_hparams)
    umap_embedding = umap_model.fit_transform(embedding)

    # Plotting
    fig, ax = plt.subplots(figsize=(7, 5))

    # Group data by labels and plot each group
    unique_labels = np.unique(label)
    for ulabel in unique_labels:
        idx = label == ulabel
        count = np.sum(idx)  # Count of rows with this label
        scatter = ax.scatter(umap_embedding[idx, 0], umap_embedding[idx, 1], s=3, label=f'{ulabel} (n={count})', alpha=0.6)

    # Customize the legend
    ax.legend(title='User IDs', markerscale=5, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set the title
    plt.title('UMAP Dimensionality Reduction', fontsize=15, fontweight='bold')
    plt.tight_layout()  # Adjust layout to make room for the legend

    if save_to_file:
        plt.savefig(f_path)
        plt.close() 
    else:
        plt.show()
