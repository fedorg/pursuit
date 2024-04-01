# make parallel requests to https://solar.furtrack.com/get/p/{0..513846}
import concurrent.futures
import json
import os

import requests


def get_url(url):
    return requests.get(
        url,
        cookies={
            "cf_chl_3": "5e14c56505bec0e",
            "cf_clearance": "vX7pPf0iCsy1JJpBeScQZnVX9jJO4N8OP66ub2mnboM-1711844877-1.0.1.1-h3qYEPgiN0mdPnDqgpElKUd49PnpajNI_.KgRoWNd2vguOeCrAxoWs0A1TT_dH78AS5a.Z.5nsVEbbZUJum4vw",
        },
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        },
    )


def make_requests(urls: list[str], max_requests: int = 20, verbose: bool = True):
    # break urls into chunks of max_requests
    url_chunks = [urls[i : i + max_requests] for i in range(0, len(urls), max_requests)]
    total = 0
    for chunk in url_chunks:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            res = executor.map(get_url, chunk)
            total += len(chunk)
            if verbose:
                print(f"Requested {total}/{len(urls)} urls")
            for r in res:
                yield r


def get_char_tags(tags: list[dict]):
    return [
        t["tagName"].removeprefix("1:")
        for t in tags
        if t.get("tagName", "").startswith("1:")
    ]


def check_single_character_tags(tags: list[dict]):
    tagNames = [t.get("tagName", "") for t in tags]
    char_tags = get_char_tags(tags)
    # print(char_tags)
    if len(char_tags) != 1:
        return False
    bad_tags = [t for t in tagNames if t in {"tagging_incomplete", "missing_character"}]
    if len(bad_tags):
        return False
    return True


# def get_full_img_url(j: dict):
#     postId = j.get("post", {}).get("postId", "")
#     postStub = j.get("post", {}).get("metaFingerprint", "")
#     userId = j.get("post", {}).get("submitUserId", "")
#     if not postId or not postStub or not userId:
#         return ""
#     return f"https://orca2.furtrack.com/gallery/{userId}/{postId}-{postStub}.jpg"


def get_img_url(j: dict):
    postId = j.get("post", {}).get("postId", "")
    if not postId:
        return ""
    return f"https://orca2.furtrack.com/thumb/{postId}.jpg"


def check_response(raw: dict, post_id: int):
    tests = [
        lambda raw: (raw.get("success") is True),
        lambda raw: (check_single_character_tags(raw.get("tags", []))),
        lambda raw: (get_img_url(raw)),
        lambda raw: (raw.get("post", {}).get("videoId", "") == ""),
    ]
    rslt = all((test(raw) for test in tests))
    if not tests[0](raw):
        print(post_id, raw)
    return rslt


def get_character(res: dict):
    tags = res.get("tags", [])
    char_tags = get_char_tags(tags)
    if len(char_tags) != 1:
        char_tags = [""]
    return char_tags[0]


def download_images(urls_names: list[tuple[str, str]], folder: str):
    os.makedirs(folder, exist_ok=True)
    # skip if file already exists
    dir_contents = os.listdir(folder)
    len_input = len(urls_names)
    urls_names = [(u, n) for u, n in urls_names if f"{n}.jpg" not in dir_contents]
    if len(urls_names) < len_input:
        print(f"Skipped downloading {len_input - len(urls_names)} images")

    img_urls = [u for u, n in urls_names]
    names = [n for u, n in urls_names]
    rs = make_requests(img_urls, verbose=False)
    for name, r in zip(names, rs):
        if r.status_code == 200:
            with open(f"{folder}/{name}.jpg", "wb") as f:
                f.write(r.content)
        else:
            print(f"Failed to download {name}.jpg")


def get_furtrack_data(post_ids: list[int]):
    urls = [f"https://solar.furtrack.com/get/p/{i}" for i in post_ids]
    jsons = [r.json() for r in make_requests(urls)]
    mapped = [
        dict(
            char=get_character(f),
            url=get_img_url(f),
            post_id=str(post_id),
            raw=f,
        )
        if check_response(f, post_id)
        else dict(
            char="",
            url="",
            post_id=str(post_id),
            raw=f,
        )
        for post_id, f in zip(post_ids, jsons)
    ]
    return mapped


# store embeddings in a database
import sqlite3

import numpy as np


def create_table():
    if os.path.exists("furtrack.db"):
        print("furtrack.db already exists")
        return
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE furtrack
                    (post_id text PRIMARY KEY, char text, url text, raw text, embedding_id integer default NULL)""")
    # unique constraint
    c.execute("CREATE UNIQUE INDEX post_id_index ON furtrack (post_id)")
    c.execute("CREATE UNIQUE INDEX u_embedding_id_index ON furtrack(embedding_id)")
    conn.commit()
    conn.close()


def store_furtrack_data(dict_list: list[dict]):
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    try:
        if len(dict_list) == 0:
            print("No data to store")
            return
        for d in dict_list:
            c.execute(
                "INSERT OR REPLACE INTO furtrack VALUES (?, ?, ?, ?, ?)",
                (
                    d["post_id"],
                    d["char"],
                    d["url"],
                    json.dumps(d["raw"]),
                    d.get("embedding_id", None),
                ),
            )
        conn.commit()
        c.execute(
            "SELECT COUNT(*) FROM furtrack UNION ALL SELECT COUNT(*) FROM furtrack WHERE url!=''"
        )
        print(f"Rows in database: {c.fetchone()[0]}, Rows with urls: {c.fetchone()[0]}")
    except sqlite3.IntegrityError:
        print(f"post_id {d['post_id']} already exists")
    finally:
        # print("Closing connection")
        conn.close()

def row2dict(row):
    if row is None:
        return None
    return {
        "post_id": row[0],
        "char": row[1],
        "url": row[2],
        "raw": json.loads(row[3]),
        "embedding_id": row[4],
    }

def recall_furtrack_data(post_id: str):
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    c.execute("SELECT * FROM furtrack WHERE post_id=?", (post_id,))
    res = c.fetchone()
    conn.close()
    return row2dict(res)

def recall_furtrack_data_by_embedding_id(embedding_id: int):
    conn = sqlite3.connect("furtrack.db")
    c = conn.cursor()
    c.execute("SELECT * FROM furtrack WHERE embedding_id=?", (embedding_id,))
    res = c.fetchone()
    conn.close()
    return row2dict(res)

# def sqlite_add_column():
#     conn = sqlite3.connect("furtrack.db")
#     c = conn.cursor()
#     c.execute("ALTER TABLE furtrack ADD COLUMN embedding_id integer default NULL")
#     c.execute("CREATE UNIQUE INDEX u_embedding_id_index ON furtrack(embedding_id)")
#     conn.commit()
#     conn.close()


def download_and_process_furtrack_data(
    post_ids: list[int], folder: str, batch_size: int
):
    ids = [p for p in post_ids if recall_furtrack_data(str(p)) is None]
    print(f"Getting {len(ids)}/{len(post_ids)} new posts")
    # process in batches
    from itertools import islice

    iter_ids = iter(ids)

    while True:
        ids = list(islice(iter_ids, batch_size))
        if not ids:
            break
        metas = get_furtrack_data(ids)
        print(f"{[m['post_id'] for m in metas if not m['url']]} failed to get metadata")
        print(f"{len([m for m in metas if m['url']])} got metadata")
        store_furtrack_data(metas)
        download_images([(m["url"], m["post_id"]) for m in metas if m["url"]], folder)


if __name__ == "__main__":
    import random

    random.seed(42)
    post_ids = [i for i in random.sample(range(549557), 20000)]
    print(len(post_ids))

    create_table()
    # print(recall_furtrack_data("537727"))
    # print(recall_furtrack_data("2"))
    download_and_process_furtrack_data(post_ids, "furtrack_images", batch_size=40)
