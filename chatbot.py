import re
import numpy as np


# Data Preprocessing
# Data load
basic_path = "./data/Cornell Movie-Dialogs Corpus"
lines = (
    open(f"{basic_path}/movie_lines.txt", encoding="utf-8", errors="ignore")
    .read()
    .split("\n")
)
conversations = (
    open(f"{basic_path}/movie_conversations.txt", encoding="utf-8", errors="ignore")
    .read()
    .split("\n")
)


# Data mapping
# Extract data
id2line = dict()
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

conversation_ids = list()
for conversation in conversations[:-1]:
    _conversation = (
        conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    )
    conversation_ids.append(_conversation.split(","))

# Create questions and answers
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i + 1]])


# Data cleanning > 대소문자 맞춤, 축약어 제거 및 변경 등
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"[\"'-()\#/@;:<>{}+=~!|.?,]", "", text)
    return text


clean_questions = [clean_text(question) for question in questions]
clean_answers = [clean_text(answer) for answer in answers]


# Remove infrequency word
# Word counting
word2count = dict()
for question in clean_questions:
    for word in question.split():
        word2count[word] = 1 if (cnt := word2count.get(word)) == None else cnt + 1

for answer in clean_answers:
    for word in answer.split():
        word2count[word] = 1 if (cnt := word2count.get(word)) == None else cnt + 1


# Word to integer & Remove infrequency word

"""
    !!! questions 와 answers 를 위환 dictionary 를 따로 만든 이유에 대해 질의 남기기 !!!
"""

threshold = 20

questionswords2int = dict()
question_word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = question_word_number
        question_word_number += 1

answerswords2int = dict()
answer_word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = answer_word_number
        answer_word_number += 1

# Add special tokens in word dictionary
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Make reverse dictionary(only answer dictionary)
answersint2words = {w_i: w for w, w_i in answerswords2int.items()}

# Add <EOS> Token in answer data
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"

# Change word 2 int in answer data
questions_into_int = list()
for question in clean_questions:
    ints = []
    for word in question.split():
        if (w_i := questionswords2int.get(word)) == None:
            ints.append(questionswords2int.get("<OUT>"))
        else:
            ints.append(w_i)
    if ints:
        questions_into_int.append(ints)

answers_into_int = list()
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if (w_i := answerswords2int.get(word)) == None:
            ints.append(answerswords2int.get("<OUT>"))
        else:
            ints.append(w_i)
    if ints:
        answers_into_int.append(ints)


## Sorting questions and answers by the length of quetions
# 강의에서 사용한 코드 (오래걸림)
# sorted_clean_questions = list()
# sorted_clean_answers = list()
# for length in range(1, 25 + 1):
#     for i in enumerate(questions_into_int):
#         if len(i[1]) == length:
#             sorted_clean_questions.append(questions_into_int[i[0]])
#             sorted_clean_answers.append(answers_into_int[i[0]])

# 새롭게 짠 코드
max_length = 25
length2questions = {i + 1: list() for i in range(max_length)}
length2answers = {i + 1: list() for i in range(max_length)}
for i, q_int in enumerate(questions_into_int):
    if (q_length := len(q_int)) <= max_length:
        length2questions[q_length].append(questions_into_int[i])
        length2answers[q_length].append(answers_into_int[i])
sorted_clean_questions = sum(length2questions.values(), [])
sorted_clean_answers = sum(length2answers.values(), [])
