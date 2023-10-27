import os
import re
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

def calculate_entropy(pickle_dir):
    top_p_values = [0.8, 0.6, 0.4, 0.2]
    model_sizes = ['125M', '350M']
    num_copies_list = [1, 5, 15, 25]

    data = {
        'model_size': [],
        'top_p': [],
        'num_copies': [],
        'average_entropy': []
    }

    for i, top_p in enumerate(top_p_values):
        for j, model_size in enumerate(model_sizes):
            pattern = f"gpt-neo-{model_size}.*sentence_probabilities_{top_p}.pkl"
            for k, fname in enumerate(sorted(os.listdir(pickle_dir))):
                if re.fullmatch(pattern, fname):
                    with open(os.path.join(pickle_dir, fname), 'rb') as f:
                        word_probabilities = pickle.load(f)

                    all_entropies = []
                    for token_distributions in word_probabilities:
                        import pdb; pdb.set_trace()
                        scores = token_distributions['scores']
                        # Calculate entropy for each token's distribution
                        token_entropies = [-sum(p * np.log2(p) for p in token_dist if p > 0) for token_dist in scores]
                        all_entropies.extend(token_entropies)

                    # Calculate average entropy for the entire text
                    if all_entropies:
                        average_entropy = sum(all_entropies) / len(all_entropies)
                        data['model_size'].append(model_size)
                        data['top_p'].append(top_p)
                        data['num_copies'].append(num_copies_list[k])
                        data['average_entropy'].append(average_entropy)

    return pd.DataFrame(data)

# Call the function
# df = calculate_entropy("/Users/luka.borec/Downloads/Archive")
df = calculate_entropy("/project/memorization/trained/")
# # Plotting
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df, x='top_p', y='average_entropy', hue='num_copies', marker='o', palette='viridis')
#
# plt.title('Effect of top_p on Average Entropy for Different Number of Copies')
# plt.xlabel('top_p Value')
# plt.ylabel('Average Entropy')
# plt.legend(title='Number of Copies', loc='upper right')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()
