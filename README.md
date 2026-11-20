# To Yap or Not to Yap

CL3.202 final project. Renny Sharma, Sanyam Asthana.

Compares how information is spread in human vs LLM text (English and Hindi) using surprisal.

## Run

    uv sync
    cd src
    uv run build_corpus.py
    uv run train_kn.py
    uv run score_corpus.py
