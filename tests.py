# %%
import random
import faiss

from pursuit import generate_embedding, get_closest_embeddings, insert_embedding, generate_embedding_from_image
from pursuit import N_DIMS
from pursuit import get_slices


# %%
import torch.nn as nn

def test_embedding():
    bird1 = generate_embedding("/Users/user/Downloads/aegialeus.jpg")# .unsqueeze(0)
    fedor1 = generate_embedding("/Users/user/Downloads/_DSC1513.jpg")
    fedor2 = generate_embedding("/Users/user/Desktop/IMG_7466.jpg")
    cat1 = generate_embedding("/Users/user/Downloads/fluxpaw.jpg")

    # compute cosine similarity
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    print(f"cos(fedor1, fedor1) = {cos(fedor1, fedor1)}")
    print(f"cos(fedor1, fedor2) = {cos(fedor1, fedor2)}")
    print(f"cos(bird1, fedor1) = {cos(bird1, fedor1)}")
    print(f"cos(bird1, fedor2) = {cos(bird1, fedor2)}")
    print(f"cos(bird1, cat1) = {cos(bird1, cat1)}")
    print(f"cos(fedor1, cat1) = {cos(fedor1, cat1)}")

# test_embedding()

# %%

# print(get_closest_embeddings(bird1.data.numpy())[0])


# %%

def test_db():
    index = faiss.IndexFlatL2(N_DIMS)
    bird1 = generate_embedding("/Users/user/Downloads/aegialeus.jpg")# .unsqueeze(0)
    fedor1 = generate_embedding("/Users/user/Downloads/_DSC1513.jpg")
    fedor2 = generate_embedding("/Users/user/Desktop/IMG_7466.jpg")
    cat1 = generate_embedding("/Users/user/Downloads/fluxpaw.jpg")

    idx_bird = insert_embedding(index, bird1.data.numpy())
    # print(idx_bird)
    # emb_bird = index.reconstruct(idx_bird)
    idx_fedor1 = insert_embedding(index, fedor1.data.numpy())
    print(idx_fedor1)
    idx_cat1 = insert_embedding(index, cat1.data.numpy())
    # print(idx_cat1)
    # idx_fedor2 = insert_embedding(index, fedor2.data.numpy())
    eids = get_closest_embeddings(index, fedor2.data.numpy(), 3)
    assert eids[0][0] == idx_fedor1

# test_db()


# %%
import matplotlib.pyplot as plt

def test_slice():
    s_cat1 = get_slices("/Users/user/Downloads/fluxpaw.jpg")
    print(s_cat1[0].shape)
    # show all slices
    for img in s_cat1:
        plt.imshow(img)
        plt.show()

    s_bird1 = generate_embedding_from_image(get_slices("/Users/user/Downloads/aegialeus.jpg")[0])
    s_fedor1 = generate_embedding_from_image(get_slices("/Users/user/Downloads/_DSC1513.jpg")[0])
    s_fedor2 = generate_embedding_from_image(get_slices("/Users/user/Desktop/IMG_7466.jpg")[0])
    s_cat1 = generate_embedding_from_image(get_slices("/Users/user/Downloads/fluxpaw.jpg")[0])

    # slc = get_slices("/Users/user/Downloads/3fursuits.jpg", False)
    slc = get_slices("/Users/user/Downloads/Convention-4K.jpg", True)
    plt.imshow(slc[1])

# test_slice()
