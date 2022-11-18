This folder contains scripts that are a mix of: (a) documentation, (b) internal Semantic Scholar scripts that won't run for anyone outside of AI2, 
(c) experimental scripts for the ablations, and (d) continuous integration scripts.

If you're not internal to AI2, here are scripts you will care about:
- `ablations.sh`: A complete list of command line commands to reproduce all of the paper's results 
- `experiment.py`: The main script used to run the experiments present in the paper
- `dump_pairwise_model.py`: An example of S2APLER usage pipeline that's easier to look at than the above two scripts. This is how the final prod model was actually made.
- `analyze_final_data.ipynb`: Code we used to improve and debug the model. It's for documentation purposes only.

*Important* notes about `dump_pairwise_model.py`: 
- It assumes that the S2AND data is in `<code root path>/data/`. If that's not the case, you'll have to modify the `"main_data_dir"` entry in `data/path_config.json`.
- If you have a small to medium amount of RAM, don't use the `--use_cache` flag. Without the cache, it'll be slower, but will not try to fit all of the feature data into memory.

Continuous integration script:
- `run_ci_locally.sh`: Runs the CI for the repo locally
