import pandas as pd
import numpy as np
import re
import pickle as pk
np.random.seed(seed=7)

data_path = './data-toxic-kaggle/train.csv'
pkl_path = './data-toxic-kaggle/toxic_comments_100.pkl'
perturbed_path = './data-toxic-kaggle/toxic_comments_100_perturbed.pkl'

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  
    string = re.sub(r"[0-9]+", " ", string)   
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " ", string) 
    string = re.sub(r"!", " ", string) 
    string = re.sub(r"\(", " ", string) 
    string = re.sub(r"\)", " ", string) 
    string = re.sub(r"\?", " ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def take_sample():
	df = pd.read_csv(data_path) 

	toxic = df.loc[df['toxic'] == 1]

	toxic_comments = toxic['comment_text'].tolist()

	print("The number of toxic comments: %d"%len(toxic_comments))

	toxic_comments_filterd = []
	for comment in toxic_comments:
		if len(comment) < 50:
			toxic_comments_filterd.append(comment)


	print("The number of comments whose lengths are less than 50: %d"%len(toxic_comments_filterd))

	np.random.shuffle(toxic_comments_filterd)

	# take 100 samples from comments with length less than 50
	toxic_comments_filterd = toxic_comments_filterd[:100]

	# clean comments
	toxic_comments_cleand = []
	for comment in toxic_comments_filterd: 
		comment_clean = clean_str(comment, True)
		toxic_comments_cleand.append(comment_clean)


	print(toxic_comments_cleand)

	with open(pkl_path, 'wb') as f:
		 pk.dump(toxic_comments_cleand, f)

def load_samples():

	with open(pkl_path, 'rb') as f:
		toxic_comments_cleand = pk.load(f)

	# I  asked Edwin to perturb the data

	with open(perturbed_path, 'rb') as f:
		toxic_comments_perturbed = pk.load(f)  

	# for i in range(100):
	# 	print("%s --> %s"%(toxic_comments_cleand[i],toxic_comments_perturbed[i]))

	return toxic_comments_cleand, toxic_comments_perturbed


