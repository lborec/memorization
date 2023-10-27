import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    for model_size in model_sizes:
        for top_p in top_p_values:
            pattern = f"gpt-neo-{model_size}.*sentence_probabilities_{top_p}.pkl"
            matching_files = sorted([f for f in os.listdir(pickle_dir) if re.fullmatch(pattern, f)])

            for fname in matching_files:
                with open(os.path.join(pickle_dir, fname), 'rb') as f:
                    word_probabilities = pickle.load(f)

                for l, token_distributions in enumerate(word_probabilities):
                    scores = token_distributions['scores']
                    token_entropies = []

                    for token_dist in scores:
                        total_prob = np.sum(token_dist.numpy()[0])
                        if total_prob == 0:  # Skip distributions with all zeros
                            continue
                        import pdb; pdb.set_trace()
                        normalized_probs = token_dist.numpy()[0] / total_prob
                        entropy = -np.sum(normalized_probs * np.where(normalized_probs > 0, np.log2(normalized_probs), 0))
                        token_entropies.append(entropy)


                    average_entropy = sum(token_entropies) / len(token_entropies)
                    data['model_size'].append(model_size)
                    data['top_p'].append(top_p)
                    data['num_copies'].append(num_copies_list[l])
                    data['average_entropy'].append(average_entropy)

    df = pd.DataFrame(data)
    df.to_pickle('entropy.pkl')
    return df

# Call the function
df = calculate_entropy("/project/memorization/trained/")

# Assuming df is your dataframe
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='top_p', y='average_entropy', hue='num_copies', style='model_size', marker='o', palette='viridis')
plt.title('Effect of top_p on Average Entropy for Different Number of Copies and Model Sizes')
plt.xlabel('top_p Value')
plt.ylabel('Average Entropy')
plt.legend(title='Legend', loc='upper right')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
