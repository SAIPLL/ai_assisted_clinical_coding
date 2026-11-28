# Prime Automation

Predicts the primary diagnosis code (PDC) of an admission.

## Training and evaluation

Run the following commands from this directory:

```bash
uv sync
uv run python prepare_prime_automation_data.py
uv run python main.py experiment=mimiciv_icd10/prime_automation_plm_icd run_id=plm_icd
uv run python main.py experiment=mimiciv_icd10/prime_automation_plm_pdc run_id=plm_pdc
```

Each run saves the model, predictions, and config under `artifacts/<run_id>/`. Precision@1 is printed during and after training. Since each test admission has exactly one PDC, Precision@1 equals accuracy.

To evaluate a trained model without training, add `trainer.epochs=0 load_model=artifacts/<run_id>` and use a new `run_id`.

To quickly test the pipeline, add `dataset.subset=32 trainer.epochs=1`. This uses 32 random admissions per split.
