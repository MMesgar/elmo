'''
required packages:
    allennlp 
    pandas

Note: 
    check the paths to data in data_helper.py
    for velmo,velmo30k, and selmo30k: Virtual env "allennlp_wuhn" should be activated, then run 
        ipython ./online_test.py velmo|velmo30k1ep|selmo30k1ep|selmo30k5ep|velmo30k5ep steffen_even|edwin|random <clean> <perturbation>

    for elmo: Virtual env "allennlp" should be activated, then run 
        ipython ./sanity_check.py elmo steffen_even|edwin|random <clean> <perturb>
'''
import embeddings_velmo as embeddings
import numpy as np
import argparse
import data_helper
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s:%(message)s')

ignore_list = [' ']
np.random.seed(seed=7)
output_path = "./data-toxic-kaggle/"

def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_similarities(loaded_emb, clean_path, perturbed_path):

    clean_sentences, perturbed_sentences = data_helper.load_samples_txt(clean_path,perturbed_path)

    n = len(clean_sentences)
    cosine_similarities = []
    for index in range(n):
        clean_sentence = clean_sentences[index]
        perturbed_sentence = perturbed_sentences[index]
        (vec_clean, vec_pert) = embeddings.get_embeddings([clean_sentence, perturbed_sentence],loaded_emb)
        cosine_similarities.append(cos_sim(vec_clean, vec_pert))

    return (clean_sentences, perturbed_sentences, cosine_similarities)


    # if perturbation_script == 'steffen_even':
    #     perturbed_path = './data-toxic-kaggle/toxic_comments_100_mm_even_p%.1f.pkl'%p # perturbed by Steffen_even script   
    #     original, perturbed = data_helper.load_samples(perturbed_path,pkl_path)
    #     logging.info("perturbation_script: steffen_even")

    #     logging.info("data loaded")
    #     output = []
    #     for index in range(len(original)):
    #         logging.info("index:%d"%index) 
    #         original_sample = original[index]
    #         perturbed_sample = perturbed[index]
    #         (vec_orig, vec_pert) = embeddings.get_embeddings([original_sample, perturbed_sample],load_emb)
    #         output.append("%s,%s,%.2f"%(original_sample,perturbed_sample, cos_sim(vec_orig, vec_pert)))


    # elif perturbation_script == 'edwin':
    #     perturbed_path = './data-toxic-kaggle/toxic_comments_100_perturbed.pkl' # perturbed by Edwin's script
    #     original, perturbed = data_helper.load_samples(perturbed_path,pkl_path)
    #     logging.info("perturbation_script: edwin")

    #     logging.info("data loaded")
    #     output = []
    #     for index in range(len(original)):
    #         logging.info("index:%d"%index) 
    #         original_sample = original[index]
    #         perturbed_sample = perturbed[index]
    #         pert = ''
    #         for i,ch in enumerate(original_sample):
    #             prob = np.random.uniform()
    #             if ( prob<= p) and (ch not in ignore_list):
    #                 # disturb
    #                 pert +=  perturbed_sample[i]
    #             else:
    #                 pert += ch
    #         #print([original_sample,pert])
    #         (vec_orig, vec_pert) = embeddings.get_embeddings([original_sample, pert],load_emb)
    #         output.append("%s,%s,%.2f"%(original_sample,pert, cos_sim(vec_orig, vec_pert)))

    # else:
    #     raise ValueError("%s is not implemented!"%perturbation_script)

    # return output

def compare_with_random(load_emb):

    pkl_path = './data-toxic-kaggle/toxic_comments_100.pkl' 

    perturbed_path = './data-toxic-kaggle/toxic_comments_100_perturbed.pkl' # perturbed by Edwin's script
    
    original, perturbed = data_helper.load_samples(perturbed_path,pkl_path)
    
    logging.info("data loaded")

    perturbed = original[:]

    np.random.shuffle(perturbed)

    logging.info("sentences are shuffled")
    
    output = []
        
    for index in range(len(original)):
            
        logging.info("index:%d"%index) 
        
        original_sample = original[index]
        
        perturbed_sample = perturbed[index]
            
        #print([original_sample, perturbed_sample])
        
        (vec_orig, vec_pert) = embeddings.get_embeddings([original_sample, perturbed_sample],load_emb)
        
        output.append("%s,%s,%.2f"%(original_sample,perturbed_sample, cos_sim(vec_orig, vec_pert)))

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", action="store",help="path to clean sentences",dest='clean_path')
    parser.add_argument("-p", action="store", help="path to perturbed sentences",dest='pert_path')
    parser.add_argument("-s", action="store", help="path to shuffled sentences",dest='shuffle_path')

    args = parser.parse_args()
    clean_path  = args.clean_path
    pert_path = args.pert_path
    shuffle_path = args.shuffle_path

    #"selmo30k5ep"
    SELMo_json_file = './SELMO.30k.5ep/selmo.30k.5ep.options.json'
    SELMo_hdf5_file = 'SELMO.30k.5ep/selmo.30k.5ep.weights.hdf5'
    
    #velmo30k5ep
    VELMo_json_file = './VELMO.30k.5ep/options.json' 
    VELMo_hdf5_file = './VELMO.30k.5ep/velmo_5_epoch.hdf5'

    loaded_SELMo =  embeddings.load_emb(SELMo_json_file, SELMo_hdf5_file)

    logging.info("SELMo loaded")

    loaded_VELMo =  embeddings.load_emb(VELMo_json_file, VELMo_hdf5_file)

    logging.info("VELMo loaded")    

    for p in [0.1,0.2,0.4,0.6,0.8,1.0]:
                
        pert_path_complete = pert_path+'_p_%.1f.txt'%p

        _, _, cosine_SELMo_clean_pert = cosine_similarities(loaded_SELMo, clean_path, pert_path_complete)

        _, _, cosine_SELMo_clean_shuffle = cosine_similarities(loaded_SELMo, clean_path, shuffle_path)

        _, _, cosine_VELMo_clean_pert = cosine_similarities(loaded_VELMo, clean_path, pert_path_complete)

        _, _, cosine_VELMo_clean_shuffle = cosine_similarities(loaded_VELMo, clean_path, shuffle_path)

        N = float(len(cosine_SELMo_clean_pert))

        count = 0
        
        diffs = []

        for (cos_s_cp, cos_v_cp) in zip(cosine_SELMo_clean_pert,cosine_VELMo_clean_pert): #_cp means _clean_pert
 
            if cos_v_cp > cos_s_cp:
 
                count += 1

            diff = cos_v_cp - cos_s_cp

            diffs.append(diff)

        # compute metrics to report
        ratio = count / N

        min_diff,max_diff = np.min(diffs),np.max(diffs)

        avg_diff, var_diff = np.average(diffs),  np.var(diffs)

        logging.info('p=%.1f: ratio = %f, min_diff = %f, max_diff = %f, avg_diff =%f, var_diff = %f '%(p,ratio,min_diff,max_diff,avg_diff, var_diff))
        
        # compare with shuffled sentence
        count_selmo_shuffle = 0

        for (cos_s_cp, cos_s_cs) in zip(cosine_SELMo_clean_pert,cosine_SELMo_clean_shuffle):#cs means clean suffle

            if cos_s_cp > cos_s_cs: 

                count_selmo_shuffle += 1

        ratio_selmo_shuffle = count_selmo_shuffle / N

        count_velmo_shuffle = 0

        for (cos_v_cp, cos_v_cs) in zip(cosine_VELMo_clean_pert,cosine_VELMo_clean_shuffle):#cs means clean suffle

            if cos_v_cp > cos_v_cs: 

                count_velmo_shuffle += 1

        ratio_velmo_shuffle = count_velmo_shuffle / N

        logging.info('p=%.1f: ratio_selmo = %.2f, ratio_velmo = %.2f'%(p,ratio_selmo_shuffle,ratio_velmo_shuffle))

        logging.info('************************')





