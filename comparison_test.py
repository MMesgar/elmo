import embeddings
import numpy as np
import argparse
import data_helper

ignore_list = [' ']
np.random.seed(seed=7)
output_path = "./data-toxic-kaggle/comparisons"

def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def compare(loaded_elmo):
    original, perturbed = data_helper.load_samples()
    print("data loaded")
    output = []
    for i in range(100):
        print(i)
        orig   = original[i]
        p_1 = perturbed[i]
        (vec_orig, vec_p_1) = embeddings.get_embeddings_elmo([orig, p_1],loaded_elmo)
        output.append("%s,%s,%.2f"%(orig,p_1, cos_sim(vec_orig, vec_p_1)))
    return output

def compare_with_probaility(load_elmo, p=1.0):
    original, perturbed = data_helper.load_samples()
    print("data loaded")
    output = []
    for index in range(len(original)):
        print(index) 
        original_sample = original[index]
        perturbed_sample = perturbed[index]
        pert = ''
        for i,ch in enumerate(original_sample):
            prob = np.random.uniform()
            if ( prob<= p) and (ch not in ignore_list):
                # disturb
                pert +=  perturbed_sample[i]
            else:
                pert += ch
        #print([original_sample,pert])
        (vec_orig, vec_pert) = embeddings.get_embeddings_elmo([original_sample, pert],loaded_elmo)
        output.append("%s,%s,%.2f"%(original_sample,pert, cos_sim(vec_orig, vec_pert)))
    return output

def online_compare(loaded_elmo, original, perturbed ):
    print("data loaded")
    output = []
    (vec_orig, vec_p_1) = embeddings.get_embeddings_elmo([original, perturbed],loaded_elmo)
    return cos_sim(vec_orig, vec_p_1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("emb", help="type of embeddings")

    args = parser.parse_args()
    emb_type = args.emb
    print(emb_type)
    # loaded_elmo =  embeddings.load_elmo()
    # print("elmo loaded")    

    # for p in [0.1, 0.2, 0.4, 0.8, 1.0]:
    #     print(p)
    #     comparions = compare_with_probaility(loaded_elmo,p)
    #     with open(output_path+"_p:%f"%p,'wt') as file:
    #         file.write("\n".join(comparions))



    # parser = argparse.ArgumentParser()
    # parser.add_argument("original", help="original sentence")
    # parser.add_argument("perturbed", help="perturbed sentence")
    # args = parser.parse_args()
    # o = args.original
    # p = args.perturbed
    # print("%s, %s, %.2f"%(o,p,online_compare(loaded_elmo, o, p)))

