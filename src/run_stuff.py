from kneyser_ney_measure import NLTKLexicalSurprisalScorer
from neural_measure import NeuralSurprisalScorer
import csv

def main():
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

            for token, surprisal in n_pairs:
                print(token, '\t', surprisal)

            print('=' * 50)

            for token, surprisal in kn_pairs:
                print(token, '\t', surprisal)

            break

if __name__ == "__main__":
    main()
            


