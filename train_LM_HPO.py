# LINK TO THE REPOSITORY ON WHICH WE RELY: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import transformers
from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed, EarlyStoppingCallback,
)
from transformers.trainer_utils import is_main_process, BestRun

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef

import wandb
"""
#######################################################
    WANDB CONFIG
#######################################################
"""
os.environ["WANDB_PROJECT"]="PRAFAI_FA_Debut"
os.environ["WANDB_LOG_MODEL"]="false"
os.environ["WANDB_WATCH"]="false"
#os.environ["WANDB__SERVICE_WAIT"] = "300"

# SWEEP CONFIG
# lr_scheduler_type: ['linear', 'constant', 'cosine_with_restarts']
# warmup_ratio: [0, 0.05, 0.1]
# weight_decay: [0, 0.001, 0.0001]
sweep_config = {
    'method': 'grid',
    'parameters': {
        'num_train_epochs': {
          'value': 100
        },
        'per_device_train_batch_size': {
            'value': 32
        },
        'per_device_eval_batch_size': {
            'value': 32
        },
        'seed': {
            'values': [1, 12, 123]
        },
        'learning_rate': {
            'values': [1e-5, 5e-6, 1e-6]
        },
        'lr_scheduler_type':{
            'values': ['linear', 'cosine_with_restarts', 'constant']
        },
        'warmup_ratio':{
            'values': [0, 0.1]
        },
        'weight_decay':{
            'value': 0
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project='PRAFAI_FA_Debut')

# ARGUMENTS NEEDED FOR TRAINING
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default="text-classification",
        metadata={"help": "The name of the task to train on is text classification"},
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )

    dev_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )

    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path where dataset cache will be saved."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    use_sliding_window: bool = field(
        default=True,
        metadata={"help": "If true it will use a sliding window in order to classify all the corpus. If False it will truncate overflowing tokens."}
    )

    stride_size: int = field(
        default=0,
        metadata={
            "help": "When sentences overflow and are added as new sentences, the amount of tokens that will be overlapped."
        }
    )

    metric_script_path: str = field(
        default='./seqeval_allMetrics.py',
        metadata={
            "help": "Path to the dataset loading script."
        }
    )

    def __post_init__(self):
        if self.train_file is None and self.dev_file is None and self.test_file is None:
            raise ValueError("Need a training, validation and test file")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.dev_file.split(".")[-1]
            assert validation_extension in ["csv", "json"], "`dev_file` should be a csv or a json file."
            test_extension = self.test_file.split(".")[-1]
            assert test_extension in ["csv", "json"], "`test_file` should be a csv or a json file."


# ARGUMENTS ABOUT THE MODEL
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # MISMO MODELO QUE EL QUE USABA IKER
    model_path: str = field(
        metadata={"help": "Path to pretrained model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    vocab_path: Optional[str] = field(
        default=None, metadata={"help": "Vocab file path for the tokenizer."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

# ARGUMENTS FOR EARLYSTOPPING
@dataclass
class EarlyStoppingArguments:
    """
    Arguments pertaining to early stopping configuration
    """
    early_stopping: bool = field(
        default=False,
        metadata={"help": "Activate early stopping. It requires save_best_model to be set and set everything to steps instead of epochs."},
    )

    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls."},
    )

    early_stopping_threshold: float = field(
        default=0.0001,
        metadata={"help": "Use with TrainingArguments metric_for_best_model and early_stopping_patience to denote how much the specified metric must improve to satisfy early stopping conditions."},
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, EarlyStoppingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args: ModelArguments
        data_args: DataTrainingArguments
        training_args: TrainingArguments
        early_stopping_args: EarlyStoppingArguments

        model_args, data_args, training_args, early_stopping_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        model_args, data_args, training_args, early_stopping_args = parser.parse_args_into_dataclasses()

    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    callbacks = []
    if early_stopping_args.early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_args.early_stopping_patience,
            early_stopping_threshold=early_stopping_args.early_stopping_threshold,
        ))

    last_checkpoint = None

    data_files = {}
    if training_args.do_train:
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        else:
            raise ValueError("Need a training file for training or hyperparameter optimization.")

    if training_args.do_eval:
        if data_args.dev_file is not None:
            data_files["dev"] = data_args.dev_file
        else:
            raise ValueError("Need a dev file for evaluating or hyperparameter optimization.")

    if training_args.do_predict:
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need a test file for predicting.")

    if data_args.train_file.endswith(".csv"):
        datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
    else:
        raise ValueError("Need a .csv files")

    if data_args.dataset_cache_dir is not None and not os.path.exists(data_args.dataset_cache_dir):
        os.makedirs(data_args.dataset_cache_dir)

    label_list = [0,1]
    num_labels = 2

    # CARGAMOS LA CONFIGURACIÓN DEL MODELO
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        hidden_dropout_prob=0.15,    #DROPOUT ADDED
        attention_probs_dropout_prob=0.15,
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_args.model_path,
            from_tf=bool(".ckpt" in model_args.model_path),
            config=config,
            cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        config=config,
        add_prefix_space=True,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    # COLUMNS WITH LABEL AND COLUMNS WITH TEXT
    text_column_name = "reports"
    label_column_name = "debut_AF"

    # SLIDING WINDOW
    def tokenize_and_align_labels(examples):
        window_size = data_args.stride_size  # int(data_args.max_seq_length / 2)
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            max_length=data_args.max_seq_length,
            truncation=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=data_args.use_sliding_window,
            stride=window_size,
            is_split_into_words=False,
        )

        if not data_args.use_sliding_window:
            tokenized_inputs['overflow_to_sample_mapping'] = list(range(0, len(tokenized_inputs['input_ids'])))
            print("TOKENIZE AND ALIGN LABELS")
            print(tokenized_inputs['overflow_to_sample_mapping'])

        for file_fragment_index in range(0, len(tokenized_inputs['input_ids'])):

            original_file_index = tokenized_inputs['overflow_to_sample_mapping'][file_fragment_index]

            file_labels = examples[label_column_name][original_file_index]

            global_attention: list[int] = [0]

            tokenized_inputs.setdefault('labels', []).append(file_labels)
            tokenized_inputs.setdefault('global_attention_mask', []).append(global_attention)

            for key in examples.keys():
                file_fragment_data = examples[key][original_file_index]
                tokenized_inputs.setdefault(key, []).append(file_fragment_data)

        return tokenized_inputs

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        batch_size=2000,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    print(datasets)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(preds, axis=1).tolist()
        true_labels = p.label_ids
        if not isinstance(true_labels, list):
            true_labels = true_labels.tolist()

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        specificity = tn / (tn + fp)
        mcc = matthews_corrcoef(true_labels, predictions)
        result = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "mcc": mcc
        }
        return result

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    output_dir_parent = training_args.output_dir

    # Initialize our Trainer
    def train(config=None):
        with wandb.init(config=config):
            config = wandb.config
            training_args.num_train_epochs = config.num_train_epochs
            training_args.per_device_train_batch_size = config.per_device_train_batch_size
            training_args.per_device_eval_batch_size = config.per_device_eval_batch_size
            training_args.learning_rate = config.learning_rate
            training_args.lr_scheduler_type = config.lr_scheduler_type
            training_args.warmup_ratio = config.warmup_ratio
            training_args.weight_decay = config.weight_decay
            training_args.seed = config.seed
            set_seed(training_args.seed)

            # create output directory for each seed
            #current_output_dir = f'{output_dir_parent}/{training_args.learning_rate}_{training_args.lr_scheduler_type}_{training_args.warmup_ratio}_{training_args.weight_decay}/{training_args.seed}/'
            #Path(current_output_dir).mkdir(parents=True, exist_ok=True)
            #training_args.output_dir = current_output_dir

            trainer = Trainer(
                model=model_init(),
                args=training_args,
                train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
                eval_dataset=tokenized_datasets["dev"] if training_args.do_eval else None,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )

            # Training
            if training_args.do_train:
                checkpoint = None
                if training_args.resume_from_checkpoint is not None:
                    checkpoint = training_args.resume_from_checkpoint
                elif last_checkpoint is not None:
                    checkpoint = last_checkpoint

                train_result = trainer.train(resume_from_checkpoint=checkpoint)

                trainer.save_model()  # Saves the tokenizer too for easy upload

                metrics = train_result.metrics
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)

                output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
                if trainer.is_world_process_zero():
                    with open(output_train_file, "w", encoding='utf8') as writer:
                        logger.info("***** Train results *****")
                        for key, value in sorted(train_result.metrics.items()):
                            logger.info(f"  {key} = {value}")
                            writer.write(f"{key} = {value}\n")

                trainer.save_state()

            def predict_and_save_to_conll(prediction_dataset: str, output_file: str, metric_key_prefix: str = 'eval'):
                if trainer.is_world_process_zero() and data_args.task_name == 'text-classification':
                    prediction_dataset = tokenized_datasets[prediction_dataset]
                    output_predictions_file = os.path.join(training_args.output_dir, output_file)

                    prediction_results = trainer.evaluate(prediction_dataset)
                    predictions, labels, _ = trainer.predict(prediction_dataset, metric_key_prefix=metric_key_prefix)

                    # Save predictions
                    merged_true_predictions = []
                    merged_true_labels = []

                    previous_file_indx = None
                    with open(f'{output_predictions_file}.conll', "w", encoding='utf8') as writer:

                        for file_oas, file_indx, prediction, label, *token_data in zip(
                                prediction_dataset['ID'],
                                prediction_dataset['overflow_to_sample_mapping'],
                                predictions, labels,
                                prediction_dataset['informes'],
                                ):

                            if file_indx != previous_file_indx:
                                previous_file_indx = file_indx

                                # Escribo
                                if previous_file_indx is not None:
                                    writer.write('\n\n')
                                writer.write(f'FILE {file_oas}\n')

                                merged_true_predictions.append([])
                                merged_true_labels.append(label)

                            writer.write(f'{label} {prediction}\n')

                            merged_true_predictions[file_indx].append(prediction)

                    for file_index, prediction in enumerate(merged_true_predictions):
                        merged_true_predictions[file_index] = np.mean(prediction, axis=0)

                    # Make sliding window do not have advantage: update prediction_results with new metrics obtained from compute_metrics
                    merged_metrics = compute_metrics(
                        EvalPrediction(
                            predictions=merged_true_predictions,
                            label_ids=merged_true_labels
                        )
                    )

                    for key in list(merged_metrics.keys()):
                        if not key.startswith(f"{metric_key_prefix}_"):
                            merged_metrics[f"{metric_key_prefix}_{key}"] = merged_metrics.pop(key)

                    prediction_results.update(merged_metrics)

                    # Log evaluation
                    logger.info("***** Eval results *****")
                    for key, value in prediction_results.items():
                        logger.info(f"  {key} = {value}")

                    # Save evaluation in json
                    with open(f'{output_predictions_file}_results.json', "w", encoding='utf8') as writer:
                        json.dump(prediction_results, writer, indent=2, ensure_ascii=False)

            if training_args.do_eval:
                logger.info("*** Evaluate ***")
                predict_and_save_to_conll(prediction_dataset='dev', output_file='eval_predictions')

            if training_args.do_predict:
                logger.info("*** Predict ***")
                predict_and_save_to_conll(prediction_dataset='test', output_file='test_predictions')

    wandb.agent(sweep_id, train, count=150)
    wandb.finish()

def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()