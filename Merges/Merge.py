import os

MODEL_SAVE_PATH = "/dtu/blackhole/"
MODEL_NANE1 = "mistralai_Mistral-7B-Instruct-v0.1"
MODEL_NAME2 = "BioMistral_BioMistral-7B"

model_path1 = os.path.join(MODEL_SAVE_PATH, MODEL_NANE1)
model_path2 = os.path.join(MODEL_SAVE_PATH, MODEL_NAME2)

MERGE_MODEL_DARE = "BioMistral-7B-mistral7instruct-dare"
yaml_config = """
models:
  - model: {model_path1}
    # No parameters necessary for base model
  - model: {model_path2}
    parameters:
      density: 0.5
      weight: 0.5
merge_method: dare_ties
base_model: {model_path1}
parameters:
  int8_mask: true
dtype: bfloat16
""".format(
    model_path1=model_path1,
    model_path2=model_path2,
)

# Save config as yaml file
with open("config_dare.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_config)

MERGE_MODEL_SLERP = "BioMistral-7B-mistral7instruct-slerp"
yaml_config = """
slices:
  - sources:
      - model: {model_path1}
        layer_range: [0, 32]
      - model: {model_path2}
        layer_range: [0, 32]
merge_method: slerp
base_model: {model_path1}
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
""".format(
    model_path1=model_path1,
    model_path2=model_path2,
)

# Save config as yaml file
with open("config_slerp.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_config)

MERGE_MODEL_TIES = "BioMistral-7B-mistral7instruct-ties"
yaml_config = """
models:
  - model: {model_path1}
  - model: {model_path2}
    parameters:
      density: 0.5
      weight: 0.5
merge_method: ties
base_model: {model_path1}
parameters:
  normalize: true
dtype: bfloat16
""".format(
    model_path1=model_path1,
    model_path2=model_path2,
)

# Save config as yaml file
with open("config_ties.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_config)

os.system(
    "mergekit-yaml config_dare.yaml {merge_model_path} --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code".format(
        merge_model_path=os.path.join(MODEL_SAVE_PATH, MERGE_MODEL_DARE)
    )
)
os.system(
    "mergekit-yaml config_slerp.yaml {merge_model_path} --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code".format(
        merge_model_path=os.path.join(MODEL_SAVE_PATH, MERGE_MODEL_SLERP)
    )
)
os.system(
    "mergekit-yaml config_ties.yaml {merge_model_path} --copy-tokenizer --cuda --low-cpu-memory --trust-remote-code".format(
        merge_model_path=os.path.join(MODEL_SAVE_PATH, MERGE_MODEL_TIES)
    )
)
