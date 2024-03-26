## Replicating the project

This project contains the code for my MSc thesis _On the Effects of Nucleus Sampling on Text Memorization in Large
Language
Models_.

### Setup

- Manually download **_openwebtext.tar.xz_**
  from [here](https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx). Unpack the dataset somewhere by
  running `xz -d [filename.xz]`; it will create a directory called `openwebtext`. (This part is not done
  programmatically because it requires Google Drive access authorization.)
- Navigate to the root of the project and run `make install` and then enter the virtual environment by
  running `source venv/bin/activate`.

### Sampling and creating duplicates

- Run`memorization sample --project_path=<path_to_project>` `--dataset_path=<path_to_dataset>`where `path_to_project`
  can be something like `"/Users/luka/memorization"` and `dataset_path` should be similarly the path to
  the `openwebtext` folder.
- Check the `memorization/memorization/cli/sample` for the rundown of what the sampler does exactly.

### Finetuning the models

- Run `memorization train --model_type=MODEL`, where `MODEL` has to be either `gpt-neo-125` or `gpt-neo-350`. This will
  finetune the model on the sampled dataset.

### Evaluating the finetuned models

- Run `memorization evaluate` to calculate perplexity on the trained models in the `trained/` folder. This is done
  post-training on a few saved checkpoints because otherwise it would forever.
- BLEU score is also calculated post-training manually with a script.

### Running experiments

- Run `memorization run_experiments --model_path MODEL_PATH'` where `MODEL_PATH` is the path to the trained model.
- This is first going to run greedy search experiments, and then experiments for nucleus sampling for 0.2, 0.4, 0.6 and
  0.8 top_p values. These are hardcoded and can be changed in the code.

### Calculating the results

- The results for the various experiments are calculated by scripts in `memorization/dataset/stats`. These should be run
  manually.
- The results are processed and visualized in the `Figures.ipynb` notebook.