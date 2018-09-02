import embeddings
import numpy as np
import argparse
import data_helper

output_path = "./data-toxic-kaggle/comparisons"

def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_parser():
    parser = argparse.ArgumentParser(
        prog='ELMO vs NNLM',
        usage='ipython elmo_test.py [sentence1] [sentence2]',
        description='This model computes the cosine similarity between two sentences',
        add_help=True
    )

    parser.add_argument("sentences", nargs=2, help="The sentences to compare cosine similarity")

    return parser

def compare():
    original, perturbed = data_helper.load_samples()
    output = []
    for i in range(100):
        orig   = original[i]
        p_1 = perturbed[i]


        (vec_orig, vec_p_1), _ = embeddings.get_embeddings_elmo_nnlm([orig, p_1])
        output.append("%s,%s,%.2f"%(orig,p_1, cos_sim(vec_orig, vec_p_1)))
    return output


if __name__ == "__main__":
    # comparions = compare()
    # print(comparions)
    # with open(output_path,'wt') as file:
    #     file.write("\n".join(comparions))

    parser = get_parser().parse_args()
    sentence1, sentence2 = parser.sentences
    print([sentence1, sentence2])
    # Get embeddings corresponding to each sentences
    results_elmo, results_nnlm = embeddings.get_embeddings_elmo_nnlm([sentence1, sentence2])

    print("[Cosine Similarity]")
    print("\"{}\" vs \"{}\"".format(sentence1, sentence2))
    print("ELMo:", cos_sim(results_elmo[0], results_elmo[1]))
    print("NNLM:", cos_sim(results_nnlm[0], results_nnlm[1]))
