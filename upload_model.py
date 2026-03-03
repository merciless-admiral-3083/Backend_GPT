from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="model_domain_tuned_new.pt",
    path_in_repo="model_domain_tuned_new.pt",
    repo_id="merciless-admiral/200M_Param_GPT", 
    repo_type="model",
)

print("Upload complete.")