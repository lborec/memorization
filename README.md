## Replicating the project

0.
    - Manually download **_openwebtext.tar.xz_**
      from [here](https://drive.google.com/drive/folders/1IaD_SIIB-K3Sij_-JjWoPy_UrWqQRdjx). Unpack the dataset somewhere; it will create a directory called `openwebtext`. (This part is not done
      programmatically because it requires Google Drive access authorization.)
    - Navigate to the root of the project and run `make install` and then enter the virtual environment by running `source venv/bin/activate`.
1.  Run `memorization sample --project_path=<path_to_project>` `--dataset_path=<path_to_dataset>`where `path_to_project` can be something like `"/Users/luka/memorization"` and `dataset_path` should be similarly the path to the `openwebtext` folder. Check the `memorization/memorization/cli/sample` for the rundown of what the sampler does exactly.
2. 

# CLI Tools
### Sampling

The sampler will sample a subset of documents according to the *Quantifying Memorization Across Neural Language Models*
paper by Carlini N, Ippolito D, Jagielski M et al.

### Training

### Finetuning