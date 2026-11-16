"""REFERENCE ONLY. Merge marked calls into YOUR original PyTorch training code."""
from mlflow_config import RunInfo
from experiment_logger import tracked_run, log_metrics, log_checkpoint, log_test_metrics

info = RunInfo(
    owner="REPLACE_WITH_DAGSHUB_USERNAME",
    environment="local",
    model_name="resnet18",  # or mobilenet / unet / custom
    dataset_version="REPLACE_WITH_ACTUAL_DATASET_VERSION",
    seed=42,
    task="classification",
)

# PSEUDOCODE ONLY: train_one_epoch, validate and model must be defined in YOUR code.
# Keep all existing training variables, loop, checkpoint selection, datasets unchanged.
# with tracked_run(info, params={"epochs": existing_epochs, "batch_size": existing_batch_size}):
#     for epoch in range(existing_epochs):
#         train_loss = train_one_epoch(...)  # ORIGINAL CODE
#         val_f1 = validate(...)             # ORIGINAL CODE
#         log_metrics({"loss": train_loss}, epoch + 1, prefix="train")
#         log_metrics({"macro_f1": val_f1}, epoch + 1, prefix="val")
#     log_checkpoint(existing_best_path)     # OPTIONAL, only if file is already saved
#     # log_test_metrics(existing_test_results, split_id="VERIFIED_TEST_SPLIT")
