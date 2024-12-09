Order of scripts to run:

- `merge_models.sh`
  - Runs `merge-kit` to merge models
- `run_truthful_predict.sh`
  - Runs `run_models_truthful.py`
    - Calls `edit_dataset.py` and `edit_dataset_with_truthful.py`
- `json_to_csv.py`
  - Generates .csv files
- `dl_project_scripts/R/truthful_results_new.qmd`
  - Creates .eps plots
  
