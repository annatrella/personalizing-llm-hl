import argparse
import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer # rouge_score from pypi
import json
import numpy as np
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to inputs",
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File path to save results to",
        required=True,
    )

    args = parser.parse_args()

    return args

def bleu_score(input_text, model_output, target_output):
    # Tokenize the input, model output, and target output into lists of words
    input_tokens = nltk.word_tokenize(input_text.lower())
    model_output_tokens = nltk.word_tokenize(model_output.lower())
    target_output_tokens = nltk.word_tokenize(target_output.lower())

    # Calculate the BLEU score with smoothing
    smooth = SmoothingFunction().method1
    bleu_score_value = sentence_bleu([target_output_tokens], model_output_tokens, smoothing_function=smooth)

    return bleu_score_value

def rouge_scores(model_output, target_output):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(target_output, model_output)
    rouge_1_score = scores['rouge1'].fmeasure
    rouge_l_score = scores['rougeL'].fmeasure

    return rouge_1_score, rouge_l_score

def calculate_scores(input_text, model_output, target_output):
    # Calculate BLEU score
    bleu_score_value = bleu_score(input_text, model_output, target_output)
    # Calculate ROUGE scores
    rouge_1_score, rouge_l_score = rouge_scores(model_output, target_output)

    return bleu_score_value, rouge_1_score, rouge_l_score

def main():
    args = parse_args()
    input_path = args.input_path
    output_file = args.output_file
    with open(input_path, "r") as file:
        data = json.load(file)
    N = len(data)
    bleu_scores = np.zeros(N)
    rouge_1_scores = np.zeros(N)
    rouge_l_scores = np.zeros(N)
    for i, item in enumerate(data):
        input_text = item["prompt"]
        model_output = item["model_output"]
        target_output = item["output"]
        bleu_score_value, rouge_1_score, rouge_l_score = calculate_scores(input_text, model_output, target_output)
        bleu_scores[i] = bleu_score_value
        rouge_1_scores[i] = rouge_1_score
        rouge_l_scores[i] = rouge_l_score

    print(f"Mean BLEU Score: {np.mean(bleu_scores):.3f}")
    print(f"Mean ROUGE-1 Score: {np.mean(rouge_1_scores):.3f}")
    print(f"Mean ROUGE-l Score: {np.mean(rouge_l_scores):.3f}")
    
    # pickling
    result = {
        'bleu_scores': bleu_scores,
        'rouge_1_scores': rouge_1_scores,
        'rouge_l_scores': rouge_l_scores
    }
    with open(output_file, 'wb') as file:
        pickle.dump(result, file)

if __name__ == "__main__":
    main()