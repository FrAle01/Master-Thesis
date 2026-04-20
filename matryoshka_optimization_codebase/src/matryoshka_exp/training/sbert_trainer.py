from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from ..data.training_data import TrainingDatasetLoader


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

        model = SentenceTransformer(
            cfg.model.model_name_or_path,
            device=cfg.execution.device,
            trust_remote_code=cfg.model.trust_remote_code,
        )

        loader = TrainingDatasetLoader(
            dataset_name=cfg.data.training_dataset_name,
            split=cfg.data.training_split,
            fmt=cfg.data.training_format,
            query_column=cfg.data.query_column,
            positive_column=cfg.data.positive_column,
            negative_column=cfg.data.negative_column,
            local_path=cfg.data.local_corpus_path,
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
        self.logger.info("Starting Sentence Transformers fine-tuning.")
        trainer.train()
        trainer.save_model(tcfg.output_model_dir)
        self.logger.info("Saved fine-tuned model to %s", tcfg.output_model_dir)
        return Path(tcfg.output_model_dir)
