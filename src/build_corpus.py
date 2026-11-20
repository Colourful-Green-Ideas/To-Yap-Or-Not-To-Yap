import os
import re
import random
import pandas as pd
from datasets import load_dataset
from huggingface_hub import list_repo_files

N = 1000
random.seed(42)

# Replace these with the Wikipedia page IDs
WIKI_CUTOFF = {"en": 72379170, "hi": 1415236}


# Split text into sentences
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?।])\s+", text)
    return [s for s in sentences if s]


# Make a prompt and its continuation
def make_passage(title, text):
    sentences = split_sentences(text)

    if len(sentences) < 6:
        return None

    prompt = title.strip() + "\n" + sentences[0] + " " + sentences[1]

    continuation = ""
    for sentence in sentences[2:]:
        if len((continuation + " " + sentence).split()) > 450:
            break
        continuation += " " + sentence

    continuation = continuation.strip()

    if len(continuation.split()) < 400:
        return None

    return prompt, continuation


# Collect good passages from a dataset
def collect(lang, genre, docs):

    # If this set was finished on an earlier run, load it instead of redoing it
    checkpoint = "checkpoints/" + lang + "_" + genre + ".tsv"

    if os.path.exists(checkpoint):
        print(lang, genre, ": already done, loading", checkpoint)
        return pd.read_csv(checkpoint, sep="\t").to_dict("records")

    print(lang, genre, ": starting (first progress line can take a minute)...")

    rows = []
    author_count = {}

    max_per_author = N // 5
    if genre != "fiction":
        max_per_author = N

    scanned = 0

    for title, text, author in docs:

        # Show progress every 200 documents
        scanned += 1
        if scanned % 200 == 0:
            print("   ", lang, genre, ": looked at", scanned, "docs, kept", len(rows), "/", N)

        # Give up if almost nothing is usable, instead of running for hours
        if scanned > N * 50:
            print("    stopping early: only", len(rows), "usable passages found")
            break

        if author_count.get(author, 0) >= max_per_author:
            continue

        passage = make_passage(title, text)

        if passage is None:
            continue

        author_count[author] = author_count.get(author, 0) + 1

        row = {
            "doc_id": f"{lang}_{genre}_{len(rows):04d}",
            "lang": lang,
            "genre": genre,
            "author": author,
            "prompt": passage[0],
            "human_cont": passage[1]
        }

        rows.append(row)

        if len(rows) == N:
            break

    print(lang, genre, ": DONE —", len(rows), "passages,", len(author_count), "authors")

    # Save finished sets only, so a short set gets retried next run
    if len(rows) == N:
        os.makedirs("checkpoints", exist_ok=True)
        pd.DataFrame(rows).to_csv(checkpoint, sep="\t", index=False)

    return rows


# Download a few files (shards) of a dataset and shuffle them.
# Shows a progress bar, and is cached on disk, so the next run is instant.
#   folder   = which subfolder of the dataset to use ("" = any)
#   shards   = which file numbers to download (each file is a few hundred MB)
#   revision = which branch of the dataset repo
def load_shards(repo, folder="", shards=[0], revision="main"):
    all_files = list_repo_files(repo, repo_type="dataset", revision=revision)

    parquet_files = []
    for f in sorted(all_files):
        if f.endswith(".parquet") and f.startswith(folder):
            parquet_files.append(f)

    chosen = []
    for i in shards:
        chosen.append(parquet_files[i % len(parquet_files)])

    urls = []
    for f in chosen:
        print("    downloading", repo, f, "(", len(parquet_files), "files in total )")
        urls.append("hf://datasets/" + repo + "@" + revision.replace("/", "%2F") + "/" + f)

    dataset = load_dataset("parquet", data_files=urls, split="train")
    dataset = dataset.shuffle(seed=42)
    return dataset


# Take a passage from the middle of a book
def middle_of(text, words=1500):
    word_list = text.split()
    start = len(word_list) // 3
    return " ".join(word_list[start:start + words])


# English news
def en_news():
    for x in load_shards("vblagoje/cc_news", shards=[0]):
        yield x["title"], x["text"], x["domain"]


# Wikipedia pages
def wiki(lang):
    # English Wikipedia has ~40 files sorted by page ID; file 10 is a mid-age chunk, all before the cutoff
    if lang == "en":
        shards = [10]
    else:
        shards = [0, 1]

    for x in load_shards("wikimedia/wikipedia", folder="20231101." + lang + "/", shards=shards):

        if int(x["id"]) <= WIKI_CUTOFF[lang]:
            text = x["text"]
            text = text.split("\nReferences")[0]
            text = text.split("\nसन्दर्भ")[0]

            yield x["title"], text, "wikipedia"


# English fiction
def en_fiction():
    # Books are big, so take 3 files; add more numbers if you get fewer than 1000 passages
    for x in load_shards("sedthh/gutenberg_english", shards=[0, 1, 2]):

        metadata = str(x["METADATA"])

        if "Fiction" not in metadata:
            continue

        author = re.search(r'"authors":\s*"([^"]*)"', metadata)
        title = re.search(r'"title":\s*"([^"]*)"', metadata)

        author_name = "unknown"
        book_title = ""

        if author:
            author_name = author.group(1)

        if title:
            book_title = title.group(1)

        text = middle_of(x["TEXT"])

        yield book_title, text, author_name


# Hindi news
def hi_news():
    dataset = load_shards("ai4bharat/IndicHeadlineGeneration", folder="hi/train/",
                          shards=[0], revision="refs/convert/parquet")

    for x in dataset:
        url = x.get("url") or ""

        # Skip NDTV because it is used for testing
        if "ndtv" in url:
            continue

        site = "unknown"

        if url.count("/") >= 2:
            site = url.split("/")[2]

        yield x["target"], x["input"], site


# Hindi fiction
def hi_fiction():
    folder = "raw/hi_fiction"

    files = []

    for author in os.listdir(folder):
        author_folder = os.path.join(folder, author)

        for filename in os.listdir(author_folder):
            files.append((author, filename))

    random.shuffle(files)

    for author, filename in files:
        path = os.path.join(folder, author, filename)

        with open(path, encoding="utf-8") as file:
            text = file.read()

        title = filename.replace(".txt", "")

        yield title, text, author


# Collect passages from all six sources
rows = []

rows += collect("en", "news", en_news())
rows += collect("en", "encyclopedia", wiki("en"))
rows += collect("en", "fiction", en_fiction())

rows += collect("hi", "news", hi_news())
rows += collect("hi", "encyclopedia", wiki("hi"))
rows += collect("hi", "fiction", hi_fiction())


# Save the corpus as a TSV file
df = pd.DataFrame(rows)
df.to_csv("corpus_prompts.tsv", sep="\t", index=False)

print("Saved corpus_prompts.tsv with", len(rows), "rows")


# Create the held-out NDTV test set
AGHI_FILE = "aghi_ndtv.csv"
AGHI_TEXT_COLUMN = "text"

if os.path.exists(AGHI_FILE):

    aghi = pd.read_csv(AGHI_FILE)
    test = []

    for i, text in enumerate(aghi[AGHI_TEXT_COLUMN].astype(str)):
        sentences = split_sentences(text)

        row = {
            "doc_id": f"hi_aghi_{i:04d}",
            "lang": "hi",
            "genre": "news_ood",
            "author": "ndtv",
            "prompt": " ".join(sentences[:2]),
            "human_cont": " ".join(sentences[2:])
        }

        test.append(row)

    test_df = pd.DataFrame(test)
    test_df.to_csv("aghi_test.tsv", sep="\t", index=False)

    print("Saved aghi_test.tsv with", len(test), "rows")