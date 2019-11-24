import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import operator
import sys
from ner import Parser
from collections import OrderedDict
import csv
from orderedmultidict import omdict

df_train = pd.read_csv('train.csv',sep=';')
df_test = pd.read_csv('train.csv', sep=';')
stemmer = SnowballStemmer("english")
#--------------------------
def preprocess(sent):
    stemmed = []
    for word in sent:
        stemmed.append(stemmer.stem(word))
    sent = nltk.pos_tag(sent)
    return sent
#----------------------------------------------
sent = preprocess(df_test["Word"].values)
grammar = r"""
  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
  PP: {<IN><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """

#------------------
cp = nltk.RegexpParser(grammar)
cs = cp.parse(sent)

#----------------------
test_sent = preprocess(df_train["Word"].values)
stemmed,postags  = zip(*test_sent)
df_train["Word"] = stemmed

df_train = df_train[df_train.Tag != "O"]
df_train = df_train[(df_train.POS == "NN") | (df_train.POS == "NNP") | (df_train.POS == "NNS") | (df_train.POS == "NNPS") |
        (df_train.POS == "CD") |
        (df_train.POS == "JJ") | (df_train.POS == "JJR") | (df_train.POS == "JJS")]

df_train["Tag"] = df_train["Tag"].replace(
    {"B-per":"per",
     "I-per":"per",
          "B-event":"event",
          "I-event":"event",
          "B-geo":"geo",
          "I-geo":"geo",
          "B-gpe":"gpe",
          "I-gpe":"gpe",
          "B-obj":"obj",
          "I-obj":"obj",
          "B-org":"org",
          "I-org":"org",
          "B-time":"time",
          "I-time":"time"
    }
)

#------------------
sum_occurence = df_train.groupby(["Word"]).count()
sum_occurence.loc["NATO"][0]

#----------------

df_probab = df_train.groupby("Word").Tag.value_counts()

#---------------
for index, value in df_probab.items():
    df_probab.loc[index] = value / sum_occurence.loc[index[0]][0]


#---------

df_probab.loc["NATO"]
df_probab.loc["NATO"]
for wat in df_probab.loc["NATO"].index:
    print(wat, df_probab.loc["NATO"][wat])

print(df_probab.head())

#-------------------------------------------
p = Parser()

p.load_models("models/")

#-------------------------------- Parsing the test file
test_sentences = []
with open("Test2NER.csv", "r") as input_file:
    lines = input_file.readlines()
    sentence = ""
    for index, line in enumerate(lines):
        words = line.replace('\n', '').split(";")
        if (len(words[0]) != 0) and (sentence is not ""):
            test_sentences.append(sentence)
            sentence = words[1]
        else:
            sentence = sentence + " " + words[1]

print(test_sentences[0])



#---------------------------------------- Preprocess test file sentences
preprocessed_sentences = []
test_trees = []
for sentence in test_sentences:
    processed = preprocess(sentence.split())
    preprocessed_sentences.append(processed)
    test_trees.append(cp.parse(processed))

#------------------------------ Create annotated test sentences both ways

dictionary_parsed_sentences = []
i = 0
spec_case = 0
for sentence, tree in zip(preprocessed_sentences, test_trees):
    sentence_dict = omdict()
    neural_sentence = ""
    for word_tuple in sentence:
        sentence_dict.add(word_tuple[0], "O")
        neural_sentence = neural_sentence + " " + word_tuple[0]

    for subtree in tree.subtrees(lambda t: t.label() == "NP"):
        probab = {"per": 0, "event": 0, "geo":0, "org":0, "obj":0, "gpe":0, "time":0}
        num_of_leaves = len(subtree.leaves())
        for word, pos in subtree.leaves():
            if word in df_probab.index:
                for tag_type in df_probab.loc[word].index:
                    probab[tag_type] += df_probab.loc[word][tag_type] / num_of_leaves
        if max(probab.items(), key=operator.itemgetter(1))[1] >= 0.7:
            for index, (word, pos) in enumerate(subtree.leaves()):
                if len(sentence_dict.allvalues(word)) == 1:
                    if index == 0:
                        sentence_dict.update([(word, "B-" + max(probab.items(), key=operator.itemgetter(1))[0].upper())])
                    else:
                        sentence_dict.update([(word, "I-" + max(probab.items(), key=operator.itemgetter(1))[0].upper())])
                else:
                    spec_case = spec_case + 1
    neural_parsed = p.predict(neural_sentence)
    for index, (word, tag) in enumerate(neural_parsed):
        new_tag = tag.replace("LOC", "GEO").replace("MISC", "ORG")
        try:

            if len(sentence_dict.allvalues(word)) == 1 and sentence_dict[word] == "O":
                sentence_dict[word] = new_tag
            else:
                if len(sentence_dict.allvalues(word)) > 1:
                    spec_case = spec_case + 1

        except:
            pass

    dictionary_parsed_sentences.append(sentence_dict)

print(dictionary_parsed_sentences[0].allitems())
print("spec",spec_case)

with open('Test2Pred2.csv', mode='w') as predfile:

    pred_writer = csv.writer(predfile, delimiter=';')
    i = 9001
    pred_writer.writerow( ["Sentences", "Word", "Predicted"] )
    for prediction in dictionary_parsed_sentences:

        is_first_element=True

        for key, value in prediction.allitems(): 
            
            if is_first_element:
                label = "Sentence: " + str(i)
                pred_writer.writerow( [label, key, value ] )
                is_first_element = False
            else:
                pred_writer.writerow( ["", key, value ] )


        i+=1


print("DONE")