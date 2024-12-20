Group 54 - 02456 Deep Learning project - Model merging

Group members:

Martynas Baltušis (s222858)(repo creator), Yayi Wang (s243559), Alexander Pittas (s230265), Patrik Suchopa (s241943)

Poster available as .pdf in the repo.

Run `FinalNotebook.ipynb` for complete process replication

Ensure that in the `Evaluation` folder `merge_models.sh` and `run_truthful_predict.sh` have desired/correct paths and env variables for conda environments and huggingface directories
 
Truthfulness evaluation summary:

- `merge_models.sh`
   - Runs `merge-kit` to merge models
  
- `run_truthful_predict.sh`
  - Runs `run_models_truthful.py`
    - Calls `edit_dataset_improved.py` to preprocess the TruthfulQA dataset
- `json_to_csv.py`
  - Generates .csv files (there were issues with using JSON in R for plots)
- `dl_project_scripts/truthful_R_launch.sh`
  - Runs `Rscript dl_project_scripts/R/truthful_plots.R`
  - Creates plots used in report (.qmd file `truthful_results_new.qmd` also exists if preferred)
  
