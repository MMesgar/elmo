import pandas as pd
import numpy as np
import re
import pickle as pk
import argparse
import os,sys
import subprocess # for running shell command from a python script
np.random.seed(seed=7)

############
##
# 
# for sampling from the training data:
# $python data_helper.py  -a take_sample -p 0.0 -n 6

# for perturbing with steffen script
# $python data_helper.py  -a perturb_steffen -p 0.1 -n 6

# for perturbing with Edwin script:
# $python data_helper.py  -a perturb_edwin   -p 0.1 -n 6



data_path = './data-toxic-kaggle/train.csv'
pkl_path = './data-toxic-kaggle/toxic_comments_1000.pkl'

perturbed_path = './data-toxic-kaggle/toxic_comments_1000_perturbed.pkl' # perturbed by Edwin's script
perturbed_path  = './data-toxic-kaggle/toxic_comments_1000_mm_even_p1.pkl' # perturbed by Steffen_even script
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

def take_sample(n):
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

	# take n samples from comments with length less than 50
	toxic_comments_filterd = toxic_comments_filterd[:n]

	# clean comments
	toxic_comments_cleand = []
	for comment in toxic_comments_filterd: 
		comment_clean = clean_str(comment, True)
		toxic_comments_cleand.append(comment_clean)


	print(toxic_comments_cleand)

	out_path = './data-toxic-kaggle/toxic_comments_%d'%(n)
	pkl_path = out_path+'.pkl'
	with open(pkl_path, 'wb') as f:
		 pk.dump(toxic_comments_cleand, f)

	txt_path = out_path+'.txt'
	with open(txt_path,'w') as f:
		f.write('\n'.join(toxic_comments_cleand))

def load_samples(perturbed_path, original_path ,verbose=False):

	with open(pkl_path, 'rb') as f:
		toxic_comments_clean = pk.load(f)

	# I  asked Edwin to perturb the data

	with open(perturbed_path, 'rb') as f:
		toxic_comments_perturbed = pk.load(f)

	if verbose == True:
		for i in range(1000):
			print("%s --> %s"%(toxic_comments_clean[i],toxic_comments_perturbed[i]))

	return toxic_comments_clean, toxic_comments_perturbed


def load_samples_txt(clean_path, perturbed_path ,verbose=False):

	with open(clean_path, 'r') as f:
		clean_sentences = f.readlines()

	with open(perturbed_path, 'r') as f:
		perturbed_sentences= f.readlines()

	assert(len(clean_sentences) == len(perturbed_sentences))
	if verbose == True:
		for i in range(len(clean_sentences)):
			print("%s --> %s"%(clean_sentences[i],perturbed_sentences[i]))

	return clean_sentences, perturbed_sentences

def convert_conll_to_pkl(txt_path,pkl_path):
	with open(txt_path, 'r') as f:
		lines = f.readlines()
	sentences = []
	sent = []
	for line in lines:
		if line != '\n':
			sent.append(line.strip())
		else:
			sentences.append(sent)
			sent = []
	out_lines = []
	for sent in sentences:
		out_lines.append(' '.join(sent))
	with open(pkl_path,'wb') as f:
		pk.dump(out_lines,f) 



if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-a",action="store",dest="action")
	parser.add_argument("-p",action="store",dest="prob")
	parser.add_argument("-n", action="store", dest="samples")


	parsed_args = parser.parse_args(sys.argv[1:])
	action = parsed_args.action
	print('action=%s'%action)
	prob = float(parsed_args.prob)
	print('perturbation_probability=%f'%prob)
	num_samples = int(parsed_args.samples)
	print('num_samples=%d'%num_samples)

	if action== 'take_sample':
		# ### extract N samples with length shorter than 50 from the training set of toxic comments
		take_sample(num_samples)

	elif action == 'perturb_edwin':
		os.chdir('../create_adversaries/')
		os.system('python3 disturb_gold_4_sanity.py ' +
				  '-p %f '%prob +
				  '-d ../sanity_check/data-toxic-kaggle/toxic_comments_%d.txt '%num_samples +
				  '-o ../sanity_check/data-toxic-kaggle/toxic_comments_%d_edwin_p_%.1f.txt '%(num_samples,prob) 
				  )
		os.chdir('../sanity_check/')

	elif action == 'perturb_steffen':
		# we  perturb by steffen even script
		os.chdir('../create_adversaries/')
		os.system('python3 disturb_input_perline_xy_oddeven_4_sanity.py ' + 
					'-e ../embeddings/efile.norm ' +
					'-p  %f '%prob + 
					'--even ' +
					'-d ../sanity_check/data-toxic-kaggle/toxic_comments_%d.txt '%num_samples +
					'--perturbations-file ../sanity_check/data-toxic-kaggle/perturbations-file.txt '+
					'-o ../sanity_check/data-toxic-kaggle/toxic_comments_%d_steffen_p_%.1f.txt '%(num_samples,prob) 
					)
		os.chdir('../sanity_check/')
	
	elif action == 'shuffle':
		
		clean_path = '../sanity_check/data-toxic-kaggle/toxic_comments_%d.txt'%num_samples

		with open(clean_path, 'r') as f:
			
			clean_sentences = f.readlines()	

		shuffled_sentences = []

		for sent in clean_sentences:

			shuffled_sentences.append(sent.strip())

		np.random.shuffle(shuffled_sentences)

		shffled_path = '../sanity_check/data-toxic-kaggle/toxic_comments_%d_shuffled.txt'%(num_samples) 
		
		with open(shffled_path, 'w') as f:

			f.write('\n'.join(shuffled_sentences))

	# ### Use the below command if you would like to convert the output of the steffen's script to pkl
	# for p in [0.1, 0.2, 0.4, 0.6, 0.8]:
	# 	convert_conll_to_pkl(txt_path= 'data-toxic-kaggle/toxic_comments_1000_mm_even_p%.1f.txt'%p,
	# 						 pkl_path= 'data-toxic-kaggle/toxic_comments_1000_mm_even_p%.1f.pkl'%p)

