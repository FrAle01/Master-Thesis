from __future__ import annotations

import json
import hashlib
from pathlib import Path

from ..config import ExperimentConfig
from ..data.training_data import TrainingDatasetLoader
from .lora_utils import apply_lora_to_sentence_transformer, save_lora_adapter
from .hub_utils import resolve_hf_token, resolve_hub_repo_id, upload_folder_to_hub


class SbertMatryoshkaFinetuner:
    """Fine-tune a SentenceTransformer model with Matryoshka-aware losses.

    This module intentionally targets Sentence Transformers because it provides
    official support for `MatryoshkaLoss` and `Matryoshka2dLoss`.
    """

    def __init__(self, config: ExperimentConfig, logger):
        self.config = config
        self.logger = logger

    def run(self) -> Path:
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses

        cfg = self.config
        tcfg = cfg.training
        finetune_strategy = tcfg.finetune_strategy.lower()
        if cfg.model.backend != "sentence_transformers":
            raise ValueError(
                "SbertMatryoshkaFinetuner supports only `sentence_transformers` backend. "
                f"Received: {cfg.model.backend}"
            )

        model = SentenceTransformer(
            cfg.model.model_name_or_path,
            device=cfg.execution.device,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        lora_target_modules = []
        if finetune_strategy == "lora":
            if not tcfg.lora.enabled:
                self.logger.warning("`training.lora.enabled` is false but strategy is `lora`; proceeding with LoRA.")
            lora_target_modules = apply_lora_to_sentence_transformer(model, tcfg.lora, logger=self.logger)
        elif finetune_strategy != "full":
            raise ValueError(f"Unsupported fine-tuning strategy: {tcfg.finetune_strategy}")

        loader = TrainingDatasetLoader(
            dataset_name=cfg.data.training_dataset_name,
            split=cfg.data.training_split,
            fmt=cfg.data.training_format,
            query_column=cfg.data.query_column,
            positive_column=cfg.data.positive_column,
            negative_column=cfg.data.negative_column,
            tevatron_positive_passages_column=cfg.data.tevatron_positive_passages_column,
            tevatron_negative_passages_column=cfg.data.tevatron_negative_passages_column,
            tevatron_passage_text_field=cfg.data.tevatron_passage_text_field,
            training_local_path=cfg.data.training_local_path,
            verbose=cfg.execution.verbose,
        )
        train_ds = loader.load()

        # The code assumes retrieval-oriented training. MultipleNegativesRankingLoss is a
        # strong default for query-document embedding fine-tuning.
        if tcfg.base_loss == "MultipleNegativesRankingLoss":
            base_loss = losses.MultipleNegativesRankingLoss(model)
        elif tcfg.base_loss == "MarginMSELoss":
            base_loss = losses.MarginMSELoss(model)
        else:
            raise ValueError(f"Unsupported base loss: {tcfg.base_loss}")

        if tcfg.use_2d_matryoshka:
            train_loss = losses.Matryoshka2dLoss(
                model,
                loss=base_loss,
                matryoshka_dims=tcfg.matryoshka_dimensions,
                n_layers_per_step=tcfg.n_layers_per_step,
            )
        elif tcfg.use_matryoshka:
            train_loss = losses.MatryoshkaLoss(
                model,
                loss=base_loss,
                matryoshka_dims=tcfg.matryoshka_dimensions,
            )
        else:
            train_loss = base_loss

        args = SentenceTransformerTrainingArguments(
            output_dir=tcfg.output_model_dir,
            num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=tcfg.per_device_train_batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.learning_rate,
            warmup_ratio=tcfg.warmup_ratio,
            weight_decay=tcfg.weight_decay,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            eval_steps=tcfg.eval_steps,
            fp16=tcfg.fp16,
            bf16=tcfg.bf16,
            max_steps=tcfg.max_steps,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            report_to=[],
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            loss=train_loss,
        )
        self.logger.info("Starting Sentence Transformers fine-tuning with strategy `%s`.", finetune_strategy)
        resume_checkpoint = self._resolve_resume_checkpoint(tcfg.output_model_dir)
        if resume_checkpoint:
            self.logger.info("Resuming fine-tuning from checkpoint: %s", resume_checkpoint)
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()
        if finetune_strategy == "lora":
            adapter_path = save_lora_adapter(
                model,
                tcfg.lora.output_adapter_dir,
                metadata={
                    "base_model_name_or_path": cfg.model.model_name_or_path,
                    "adapter_name": tcfg.lora.adapter_name,
                    "target_modules": lora_target_modules,
                    "matryoshka_dimensions": tcfg.matryoshka_dimensions,
                    "loss_type": type(train_loss).__name__,
                    "training_config_sha256": self._training_config_sha256(),
                },
            )
            upload_payload = self._upload_if_enabled(
                artifact_path=adapter_path,
                finetune_strategy=finetune_strategy,
            )
            self._persist_hub_upload_metadata(adapter_path, upload_payload)
            self.logger.info("Saved LoRA adapter to %s", adapter_path)
            return adapter_path

        trainer.save_model(tcfg.output_model_dir)
        artifact_path = Path(tcfg.output_model_dir)
        upload_payload = self._upload_if_enabled(
            artifact_path=artifact_path,
            finetune_strategy=finetune_strategy,
        )
        self._persist_hub_upload_metadata(artifact_path, upload_payload)
        self.logger.info("Saved fine-tuned model to %s", tcfg.output_model_dir)
        return artifact_path

    def _resolve_resume_checkpoint(self, output_dir: str) -> str | None:
        tcfg = self.config.training
        if tcfg.resume_from_checkpoint:
            return str(tcfg.resume_from_checkpoint)
        if not tcfg.auto_resume_from_last_checkpoint:
            return None
        root = Path(output_dir)
        if not root.exists():
            return None
        candidates = [p for p in root.glob("checkpoint-*") if p.is_dir()]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])

    def _training_config_sha256(self) -> str:
        payload = {
            "base_loss": self.config.training.base_loss,
            "use_matryoshka": self.config.training.use_matryoshka,
            "use_2d_matryoshka": self.config.training.use_2d_matryoshka,
            "matryoshka_dimensions": self.config.training.matryoshka_dimensions,
            "n_layers_per_step": self.config.training.n_layers_per_step,
            "epochs": self.config.training.epochs,
            "per_device_train_batch_size": self.config.training.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
            "learning_rate": self.config.training.learning_rate,
            "warmup_ratio": self.config.training.warmup_ratio,
            "weight_decay": self.config.training.weight_decay,
            "fp16": self.config.training.fp16,
            "bf16": self.config.training.bf16,
            "training_dataset_name": self.config.data.training_dataset_name,
            "training_split": self.config.data.training_split,
            "training_format": self.config.data.training_format,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _upload_if_enabled(self, artifact_path: Path, finetune_strategy: str) -> dict | None:
        hub_cfg = self.config.training.hub
        if not hub_cfg.enabled:
            return None
        token = resolve_hf_token(hub_cfg.token_env)
        repo_id = resolve_hub_repo_id(
            hub_cfg=hub_cfg,
            experiment_name=self.config.execution.experiment_name,
            finetune_strategy=finetune_strategy,
        )
        result = upload_folder_to_hub(
            folder_path=artifact_path,
            repo_id=repo_id,
            private=hub_cfg.private,
            token=token,
            commit_message=(
                f"Upload {finetune_strategy} artifact for "
                f"{self.config.execution.experiment_name}"
            ),
        )
        self.logger.info(
            "Uploaded `%s` artifact to Hugging Face repo `%s` (revision `%s`).",
            finetune_strategy,
            result["repo_id"],
            result["revision"],
        )
        return result

    def _persist_hub_upload_metadata(self, artifact_path: Path, upload_payload: dict | None) -> None:
        if not upload_payload:
            return
        metadata_path = artifact_path / "hub_upload_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(upload_payload, f, indent=2, sort_keys=True)
