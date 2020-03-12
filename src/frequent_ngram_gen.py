import time
import argparse
import os
import tokens_corpus
import phrases_corpus
import pickle

curr_dir = os.getcwd()


def parse_args():

    parser = argparse.ArgumentParser(description='Clinical Phrase Extraction')
    parser.add_argument('--input_file', type=str, default=curr_dir + '/data/input.txt', help='Name and Location of the text corpus')
    parser.add_argument('--length', type=int, default=6, help='Maximum length of N-Grams extracted')
    parser.add_argument('--frequency', type=int, default=10, help='Minimum frequency threshold of N-Grams')
    parsed_args = parser.parse_args()
    return parsed_args


print("Tokenizing...")
args = parse_args()
start_time = time.time()
token_obj = tokens_corpus.Token(args.input_file)
end_time = time.time()
print("Total time for tokenizing:", end_time - start_time, " seconds")

with open(curr_dir + "/tmp/token_dict.pkl", "wb") as token_dict_file:
    pickle.dump(token_obj, token_dict_file)

print("Extracting frequent phrases...")
start_time = time.time()
phrase_obj = phrases_corpus.Phrase(token_obj, args.length, args.frequency)
end_time = time.time()
print("Total time for extracting phrases:", end_time - start_time, " seconds")

with open(curr_dir + "/tmp/phrase_dict.pkl", "wb") as phrase_dict_file:
    pickle.dump(phrase_obj, phrase_dict_file)
