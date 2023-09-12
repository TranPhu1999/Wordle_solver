# import
import os
import json
import logging
import requests
import argparse
import numpy as np
from math import log
import cProfile, pstats
from itertools import product
from scipy.stats import entropy
from tqdm import tqdm as ProgressDisplay
from english_words import get_english_words_set
from requests.packages.urllib3.exceptions import InsecureRequestWarning

parser = argparse.ArgumentParser(description='Argument for wordle solver')
parser.add_argument("--pattern_matrix", help="If 1, create pattern matrix for all text size", default=0)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

# Disable InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Global variable
web2lowerset = get_english_words_set(['web2'], lower=True, alpha =False) # gcide / web2
web2lowerset = list(web2lowerset)

PATTERN_MATRIX_PATH = "."
INITIAL_GUESS_PATH = "."
TEXT_BASKET_PATH = "."
CORRECT, PRESENT, ABSENT = np.uint8(2), np.uint8(1), np.uint8(0)

def get_text_basket_with_size() -> dict:
  text_basket_with_size = dict()
  text_basket_file = os.path.join(TEXT_BASKET_PATH,"text_basket.json")
  if not os.path.exists(text_basket_file):
    for text in web2lowerset:
      if len(text) in text_basket_with_size:
        text_basket_with_size[len(text)].append(text)
      else:
        text_basket_with_size[len(text)] = [text]

    with open(text_basket_file, "w") as json_file:
      json.dump(text_basket_with_size, json_file)
    with open(text_basket_file, "r") as json_file:
      text_basket_with_size = json.load(json_file)
  else:
    with open(text_basket_file, "r") as json_file:
      text_basket_with_size = json.load(json_file)
  
  return text_basket_with_size

TEST_BASKET_WITH_SIZE = get_text_basket_with_size()

def words_to_int_arrays(words: str) -> list:
    return np.array([[ord(c)for c in w] for w in words], dtype=np.uint8)

# https://github.com/3b1b/videos/blob/870a6cbf30938793f93a2c9235c82bdeed31c7c6/_2022/wordle/simulations.py#L104
def create_pattern_matrix(allowed_words_list_1: list[str], allowed_words_list_2: list[str]) -> np.ndarray:
  """
  A pattern for two words represents the wordle-similarity
  pattern (grey -> 0, yellow -> 1, green -> 2) but as an integer
  between 0 and 3^5. Reading this integer in ternary gives the
  associated pattern.

  This function computes the pairwise patterns between two lists
  of words, returning the result as a grid of hash values. Since
  this can be time-consuming, many operations that can be are vectorized
  (perhaps at the expense of easier readibility), and the the result
  is saved to file so that this only needs to be evaluated once, and
  all remaining pattern matching is a lookup.
  """

  text_size = len(allowed_words_list_1[0])
  matrix_dimension_1 = len(allowed_words_list_1)
  matrix_dimension_2_big = len(allowed_words_list_2)

  # Modify: handle word list that have size too big and cause out of ram memory by split it into small list so that the pattern_matrix size don't exceed 10.000^2
  allowed_words_arr_2_big = []
  limit_matrix_dimension_2 = matrix_dimension_2_big
  if matrix_dimension_1 * matrix_dimension_2_big > 5000**2:
    limit_matrix_dimension_2 = 5000**2 // matrix_dimension_1
    for i in range(0,matrix_dimension_2_big,limit_matrix_dimension_2):
      allowed_words_arr_2_big.append(allowed_words_list_2[i:i+limit_matrix_dimension_2])
  else:
    allowed_words_arr_2_big.append(allowed_words_list_2)
  
  # Modify: For each small words list, calculate pattern matrix and then concat the result afterward
  concat_pattern_matrix = []
  for allowed_words_arr_2_small in allowed_words_arr_2_big:
    matrix_dimension_2 = len(allowed_words_arr_2_small)
    allowed_words_arr_1, allowed_words_arr_2_small = map(words_to_int_arrays,(allowed_words_list_1, allowed_words_arr_2_small))

    # equality_grid keeps track of all equalities between all pairs
    # of letters in words. Specifically, equality_grid[a, b, i, j]
    # is true when words[i][a] == words[b][j]
    equality_grid = np.zeros((matrix_dimension_1, matrix_dimension_2, text_size, text_size), dtype=bool)

    for i, j in product(range(text_size), range(text_size)):
        equality_grid[:, :, i, j] = np.equal.outer(allowed_words_arr_1[:, i], allowed_words_arr_2_small[:, j])
    # full_pattern_matrix[a, b] should represent the 5-color pattern
    # for guess a and answer b, with 0 -> grey, 1 -> yellow, 2 -> green
    full_pattern_matrix = np.zeros((matrix_dimension_1, matrix_dimension_2, text_size), dtype=np.uint8)

    # Yellow pass
    for i, j in product(range(text_size), range(text_size)):
        matches = equality_grid[:, :, i, j].flatten()
        full_pattern_matrix[:, :, i].flat[matches] = PRESENT

    # Green pass
    for i in range(text_size):
        matches = equality_grid[:, :, i, i].flatten()  # matches[a, b] is true when words[a][i] = words[b][i]
        full_pattern_matrix[:, :, i].flat[matches] = CORRECT

    # Rather than representing a color pattern as a lists of integers,
    # store it as a single integer, whose ternary representations corresponds
    # to that list of integers.
    pattern_matrix = np.dot(
        full_pattern_matrix,
        (3**np.arange(text_size)).astype(np.uint8)
    )

    if len(concat_pattern_matrix) == 0:
      concat_pattern_matrix = pattern_matrix
    else:
      concat_pattern_matrix = np.append(concat_pattern_matrix,pattern_matrix,axis=1)

  return concat_pattern_matrix

def generate_full_pattern_matrix(save_file_name: str, allowed_text: list[str]) -> np.ndarray:
    """Generating pattern matrix for text size of {len(allowed_text[0])}, length of word list is {len(allowed_text)}
    This will take a few minutes but will need to run only one time with each text size"""

    logging.info(f"""Generating pattern matrix for text size of {len(allowed_text[0])}, length of word list is {len(allowed_text)}
    This will take a few minutes but will need to run only one time with each text size""")
    pattern_matrix = create_pattern_matrix(allowed_text, allowed_text)
    # Save to file
    np.save(save_file_name, pattern_matrix)
    return pattern_matrix

def get_pattern_matrix(length: int, list_word: list[str]) -> np.ndarray:
  pattern_file = os.path.join(PATTERN_MATRIX_PATH,f"pattern_matrix_size_{length}.npy")
  if not os.path.exists(pattern_file):
    pattern_matrix = generate_full_pattern_matrix(pattern_file ,list_word)
  else:
    pattern_matrix = np.load(pattern_file)

  return pattern_matrix

def check_answer(guess_word: str, seed: int, text_size: int) -> list:
  # Check answer from API and return next pattern
  query = f"https://wordle.votee.dev:8000/random?guess={guess_word}&seed={seed}&size={text_size}"
  r = requests.get(query, verify = False)
  result = [check["result"] for check in r.json()]
  return result

def calculate_entropy(word_index, text_size: int, pattern_matrix: np.ndarray)-> float:
  """
  Return entropy of the correspond word, check the details explanation here: https://youtu.be/v68zYyaEmEA
  For a particular word, check for all possible pattern and their probability
  Then calculate the entropy for each of these word with formular of information theory E = sum(px*log(1/px))
  """

  list_probability = np.zeros(3**text_size)
  for i in range(3**text_size):
    word_basket = np.where(pattern_matrix[word_index]==i)[0].tolist()
    if len(word_basket) > 0:
      list_probability[i] = len(word_basket)/len(pattern_matrix)
  entropy_result = entropy(list_probability,base=2)
  return entropy_result

def get_best_guess(allowed_words_list: list, text_size: int, pattern_matrix: np.ndarray)-> list:
  """
  From the allow word list, get the word that give the highest entropy
  higher entropy mean it wil help split the possibility word even smaller
  """
  highest_entropy = 0
  result_index = 0
  for i, word in enumerate(ProgressDisplay(allowed_words_list)):
    entropy = calculate_entropy(i, text_size, pattern_matrix)
    if entropy > highest_entropy:
      highest_entropy = entropy
      result_index = i

  return result_index

def wordle_solver(seed: int, text_size: int) -> str:
  # main function solve wordle
  correct_pattern_int = 3**text_size - 1

  allowed_words_list = TEST_BASKET_WITH_SIZE[str(text_size)]
  pattern_matrix = get_pattern_matrix(text_size ,allowed_words_list)

  inital_guess_dict = dict()
  inital_guess_file = os.path.join(INITIAL_GUESS_PATH,"initial_guess.json")
  if os.path.exists(inital_guess_file):
    with open(inital_guess_file, "r") as json_file:
      inital_guess_dict = json.load(json_file)
      try:
        inital_guess_index = allowed_words_list.index(inital_guess_dict[str(text_size)])
      except:
        logging.info(f"Finding optimal inital guess for text size of {text_size}, this may take some time but only need to run once")
        inital_guess_index = get_best_guess(allowed_words_list, text_size, pattern_matrix)
        inital_guess_dict[str(text_size)] = allowed_words_list[inital_guess_index]
        with open(inital_guess_file, "w") as json_file:
          json.dump(inital_guess_dict, json_file)
  else:
    logging.info(f"Finding optimal inital guess for text size of {text_size}, this may take some time but only need to run once")
    inital_guess_index = get_best_guess(allowed_words_list, text_size, pattern_matrix)
    inital_guess_dict[str(text_size)] = allowed_words_list[inital_guess_index]
    with open(inital_guess_file, "w") as json_file:
      json.dump(inital_guess_dict, json_file)

  while True:
    guess_pattern = check_answer(allowed_words_list[inital_guess_index], seed, text_size)

    logging.info(f"initial guess: {allowed_words_list[inital_guess_index]}")
    logging.info(f"guess_pattern: {guess_pattern}")
    pattern_respone_dict = {"absent": ABSENT, "present": PRESENT, "correct": CORRECT}
    pattern_int = np.dot(
        [pattern_respone_dict[check] for check in guess_pattern],
        (3**np.arange(text_size)).astype(np.uint8)
    )
    if len(allowed_words_list) == 1 and pattern_int != correct_pattern_int:
      return "Can not find word!"
    if pattern_int == correct_pattern_int:
      break
    remain_words_indexes = np.where(pattern_matrix[inital_guess_index]==pattern_int)[0].tolist()
    pattern_matrix = pattern_matrix[remain_words_indexes,:][:,remain_words_indexes]
    allowed_words_list = [allowed_words_list[index] for index in remain_words_indexes]
    inital_guess_index = get_best_guess(allowed_words_list, text_size, pattern_matrix)


  return allowed_words_list[inital_guess_index]

if __name__ == "__main__":
  if args.pattern_matrix == 1:
    logging.info("Create pattern matrix for all text size")
    for length, list_word in ProgressDisplay(TEST_BASKET_WITH_SIZE.items()):
      pattern_file = os.path.join(PATTERN_MATRIX_PATH,f"pattern_matrix_size_{length}.npy")
      if not os.path.exists(pattern_file):
        generate_full_pattern_matrix(pattern_file ,list_word)

  while True:
    seed = input("Input seed: ")
    while True:
      try:
        seed = int(seed)
        break
      except:
        logging.info("\nseed has to be int! ")
        seed = input("Input seed: ")

    text_size = input("Input text size: ")
    while True:
      try:
        text_size = int(text_size)
        break
      except:
        if not isinstance(text_size,int):
          logging.info("\ntext size has to be int! ")
          text_size = input("Input text size: ")

    word_ans = wordle_solver(seed,text_size)
    logging.info(f"Word: {word_ans}")

    again = input("Continue? Yes if 1, No if not 1: ")
    if again != "1":
      break
    
  
  # profiler = cProfile.Profile()
  # profiler.enable()
  # wordle_solver(1234,5)
  # profiler.disable()
  # stats = pstats.Stats(profiler).sort_stats('tottime')
  # stats.print_stats(10)