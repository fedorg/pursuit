# %%
# %pip install ultralytics
# %pip install faiss-cpu
# %pip install torch torchvision

import random
import numpy as np
import faiss
import os

import torch
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.models as models

# print(*model.children())
N_DIMS = 25088

def generate_embedding(img_path: str):
    img = read_image(img_path)
    print(f"Generating embedding for {img_path}")
    return generate_embedding_from_image(img)

def generate_embedding_from_image(img):
    # import faulthandler; faulthandler.enable()
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # remove the last layers
    # model_embed = nn.Sequential(*list(model.children())[:-1])
    model_embed = model.features
    # print(model_embed)
    model_embed.eval()
    # convert to tensor first
    if type(img) != torch.Tensor:
        img = T.ToTensor()(img)
    img = img.float() / 255.0
    img = T.Resize((224, 224))(img)
    img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    # emb = model_embed(img.unsqueeze(0)) # add a batch dimension if using whole model
    emb = model_embed(img)
    # torch.Size([512, 7, 7])
    output = emb.flatten()
    # torch.Size([25088])
    return output


# %%

def load_embedding_db(n_dims: int) -> faiss.IndexFlatL2:
    if os.path.exists("faiss.index"):
        print("read faiss.index")
        index = faiss.read_index("faiss.index")
        print(f"ntotal: {index.ntotal}")
    else:
        index = faiss.IndexFlatL2(n_dims)
        save_embedding_db(index)
    return index

def save_embedding_db(index: faiss.Index):
    print("write faiss.index")
    faiss.write_index(index, "faiss.index")

# returns the embedding_id field for sqlite
def insert_embedding(index: faiss.Index, embedding: np.ndarray) -> int:
    # convert to float32 and add dimension
    vectors = np.array([embedding], dtype=np.float32)
    idx = index.ntotal # only add 1
    index.add(vectors)
    return idx

def get_closest_embeddings(
    index: faiss.Index, embedding: np.ndarray, top: int = 3
) -> list[int]:
    vectors = np.array([embedding], dtype=np.float32)
    distances, indices = index.search(vectors, k=top)
    return list(indices[0])

# %%

def vectorize_image(image_path: str, embed_folder: str):
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found")
        return None
    post_id = os.path.splitext(os.path.basename(image_path))[0]
    emb_path = f"{embed_folder}/{post_id}.bin"
    if os.path.exists(emb_path):
        print(f"Embedding {post_id} already exists")
        embedding = np.fromfile(emb_path, dtype=np.float32)
    else:
        print(f"Vectorizing {post_id}...")
        os.makedirs(embed_folder, exist_ok=True)
        embedding = generate_embedding(image_path).data.numpy()
        with open(f"{embed_folder}/{post_id}.bin", "wb") as f:
            f.write(embedding.tobytes()) # vectorized
    return embedding

from download import store_furtrack_data

def index_post(index: faiss.Index, metas: list[dict], image_folder: str, embed_folder: str, save_every: int = 1, total=0):
    for i, m in enumerate(metas):
        post_id = m["post_id"]
        print(f"processing {post_id} ({i}/{total})")
        embedding = vectorize_image(f"{image_folder}/{post_id}.jpg", embed_folder)
        if embedding is None:
            print(f"Failed to vectorize {post_id}")
            continue
        embedding_id = insert_embedding(index, embedding)
        print(f"embedding_id: {embedding_id}")
        m["embedding_id"] = embedding_id
        store_furtrack_data([m]) # write to database
        # store index in a file
        # https://github.com/facebookresearch/faiss/blob/main/demos/demo_ondisk_ivf.py
        # index.train(xt)
        if i % save_every == 0:
            save_embedding_db(index)

# %%
# get horizontal slices of each object
from PIL import Image
import numpy as np
# from ultralytics import YOLO

# def get_slices(img_path: str, crop = True, exclude_cars = True) -> list[np.ndarray]:
#     model = YOLO("yolov8n-seg.pt")
#     results = model(img_path)
#     xyxy = results[0].boxes.xyxy
#     boxes = results[0].boxes
#     # print(dir(results[0]))
#     img = Image.open(img_path)
#     slices = []
#     if len(xyxy) == 0:
#         return [np.array(img)]
#     # sort boxes by largest width
#     box_coords = sorted(zip(boxes, xyxy), key=lambda bc: bc[1][2] - bc[1][0], reverse=True)
#     for box, coords in box_coords:
#         name = model.names[int(box.cls)]
#         print(name)
#         if exclude_cars and name in ["car", "truck", "bus"]:
#             continue
#         x1, y1, x2, y2 = map(int, coords.floor())
#         # get image size
#         w, h = img.size
#         if crop:
#             cropped = img.crop((x1, y1, x2, y2))
#         else:
#             cropped = img.crop((x1, 0, x2, h))
#         # plt.imshow(cropped)
#         # plt.show()
#         slices.append(np.array(cropped))
#     return slices

# %%

from download import recall_furtrack_data_by_embedding_id

def get_closest_rows(index: faiss.Index, query: np.ndarray, n: int = 3) -> list[dict]:
    eids = get_closest_embeddings(index, query, n)
    print(f"closest eids: {eids}")
    # return top n rows
    rows = [recall_furtrack_data_by_embedding_id(eid) for eid in eids]
    rows = [r for r in rows if r is not None]
    print(f'closest posts: {[r["post_id"] for r in rows]}')
    return rows
    # load_embedding = lambda b: np.frombuffer(b, dtype=np.float32)
    # sort by similarity
    # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    # sorted_res = sorted(
    #     res, key=lambda r: np.dot(load_embedding(r[2]), query), reverse=True
    # )
    # return [r for r in sorted_res[:n]]


def get_closest_to_file(index: faiss.Index, path: str, n: int = 5) -> list[dict]:
    query = generate_embedding(path).data.numpy()
    return get_closest_rows(index, query, n)

def detect_characters(path: str, n: int) -> list[dict]:
    index = load_embedding_db(N_DIMS)
    query = generate_embedding(path).data.numpy()
    return get_closest_rows(index, query, n)

# %%

def batch_vectorize_images(image_folder: str, embed_folder: str):
    print("vectorizing...")
    os.makedirs(image_folder, exist_ok=True)
    files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
    for i, f in enumerate(files):
        print(f"vectorizing {i}/{len(files)}")
        vectorize_image(f"{image_folder}/{f}", embed_folder)

from download import recall_furtrack_data_by_id

def batch_reindex_embeddings(embed_folder: str):
    print("reindexing...")
    os.makedirs(embed_folder, exist_ok=True)
    emb_files = [f for f in os.listdir(embed_folder) if f.endswith(".bin")]
    post_ids = (os.path.splitext(f)[0] for f in emb_files)
    os.path.exists("faiss.index") and os.remove("faiss.index")
    index = load_embedding_db(N_DIMS)
    metas = (recall_furtrack_data_by_id(pid) for pid in post_ids)
    metas = (m for m in metas if m is not None and m["char"])
    index_post(index, metas, image_folder, embed_folder, save_every=100, total=len(emb_files))
    save_embedding_db(index)
    # print(len(metas))


if __name__ == "__main__":
    # exit on sigint
    import signal
    import os
    import sys

    def signal_handler(sig, frame):
        print("Exiting...")
        sys.exit()
        os._exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    image_folder, embed_folder = "furtrack_images", "embeddings"
    # batch_vectorize_images(image_folder, embed_folder)
    batch_reindex_embeddings(embed_folder)

    exit(0)