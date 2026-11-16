from kneyser_ney_measure import NLTKLexicalSurprisalScorer
from neural_measure import NeuralSurprisalScorer
import csv
import json

def main():
    output = []
    print("Intializing Neural Surprisal Scorer...")
    neural_model_scorer = NeuralSurprisalScorer("unsloth/Llama-3.2-1B")

    print("Intializing NLTK Lexical Surprisal Scorer...")
    nltk_model_scorer = NLTKLexicalSurprisalScorer("bigscience/bloom-7b1", order=3)
    nltk_model_scorer.load("nltk_3gram_en.pkl")

    with open("corpus_prompts.tsv", "r") as prompts_tsv:
        print("Reading the TSV...")
        csv_reader = csv.reader(prompts_tsv, delimiter='\t')

        next(csv_reader)
        for line in csv_reader:
            doc_id, lang, genre, author, prompt, human_cont = line

            print("Scoring using Neural Scorer...")
            n_tokens, n_surprisal = neural_model_scorer.score_text(prompt)

            print("Scoring using NLTK Scorer...")
            kn_tokens, kn_surprisal = nltk_model_scorer.score_text(prompt)

            n_pairs = zip(n_tokens, n_surprisal)
            kn_pairs = zip(kn_tokens, kn_surprisal)

            n_pairs_json = []
            kn_pairs_json = []

            for token, surprisal in n_pairs:
                print(token, '\t', surprisal)
                n_pairs_json.append((token, surprisal))

            print('=' * 50)

            for token, surprisal in kn_pairs:
                print(token, '\t', surprisal)
                kn_pairs_json.append((token, surprisal))

            print('\n' * 3)

            entry_dict = {
                    "doc_id": doc_id,
                    "lang": lang,
                    "genre": genre,
                    "author": author,
                    "prompt": prompt,
                    "human_cont": human_cont,

                    "nltk_pairs": kn_pairs_json,
                    "neural_pairs": n_pairs_json
                }

            output.append(entry_dict)

    with open("output.json", "w") as json_output:
        json.dump(entry_dict, json_output, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
            


