import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np

def visualize_side_by_side(pickle_dir):
    # LaTeX setup
    latex_path = "/usr/local/texlive/2023/bin/universal-darwin"
    os.environ["PATH"] += os.pathsep + latex_path
    plt.style.use('bmh')
    plt.rcParams.update({'font.size': 18, 'text.usetex': True})
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))

    top_p_values = [0.8, 0.6, 0.4, 0.2]
    model_sizes = ['125M', '350M']

    for i, top_p in enumerate(top_p_values):
        for j, model_size in enumerate(model_sizes):
            ax = axes[i][j]

            # Find corresponding pickle file
            pattern = f"gpt-neo-{model_size}.*_word_probabilities_{top_p}.pkl"
            for fname in os.listdir(pickle_dir):
                if re.fullmatch(pattern, fname):
                    with open(os.path.join(pickle_dir, fname), 'rb') as f:
                        import pdb; pdb.set_trace()
                        word_probabilities = pickle.load(f)
                        num_copies_list = [1,5,15,25]

                    for k, word_probs in enumerate(word_probabilities):
                        if not word_probs:  # Skip empty lists
                            continue
                        x = list(range(1, len(word_probs) + 1))
                        y = word_probs

                        ax.plot(x, y, label=f"Num Copies: {num_copies_list[k]}", color=f"C{k}", linewidth=0.8)

                    ax.axvline(x=250, color='r', linestyle='--')

            ax.set_ylim([-0.05, 1.05])
            ax.set_title(f"Model: {model_size}, top_p: {top_p}")
            if j == 0:
                ax.set_ylabel("Token probability")
            if i == 3:
                ax.set_xlabel("Token position")
            if i == 0 and j == 1:
                ax.legend()

    plt.tight_layout()
    plt.savefig('side_by_side_visualizations.png')
    plt.close(fig)

visualize_side_by_side("/Users/luka.borec/Downloads/pickles")
