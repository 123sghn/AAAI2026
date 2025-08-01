import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.getcwd())

import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import FinetuneCoTDataset
from src.functions import *
from src.evaluator import Evaluator

# ----- Main script ----- #
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_key", type=str, default="SAS")
parser.add_argument("--teacher_model", type=str, default="gpt_4o_mini")
parser.add_argument("--test_batch_size", type=int, default=24)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ft_lr", type=float, default=3e-4)
parser.add_argument("--generate_max_length", type=int, default=512)
args = parser.parse_args()

# ----- Function to evaluate on any dataset (val/test) ----- #
def evaluate(model, dataloader, tokenizer, evaluator, device, generation_kwargs):
    model.eval()
    raw_predictions = []
    outputs_to_decode = []

    with torch.no_grad():
        for samples in tqdm(
            dataloader,
            total=len(dataloader),
            bar_format="{l_bar}{bar:25}{r_bar}",
            desc="Evaluating",
            ascii=True,
        ):
            input_ids = samples["input_ids_l"].to(device)
            
            generate_inputs = {
                "input_ids": samples["input_ids_l"].to(device),
                "attention_mask": samples["attention_mask_l"].to(device),
                "pixel_values": samples["pixel_values"].to(device),
                "image_grid_thw": samples["image_grid_thw"].to(device)
            }

            generated_ids = model.generate(**generate_inputs, max_new_tokens=generation_kwargs["max_length"]).detach()
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
            outputs_to_decode.extend(generated_ids_trimmed)

            output_texts = tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw_predictions.extend(output_texts)

    raw_answers = [s["label"] for s in dataloader.dataset]
    evaluations, c_pred, weighted_f1, macro_f1, conf_matrix = return_evaluations_in_boolean(
        evaluator, raw_predictions, raw_answers, return_cleansed_predictions=True
    )
    accuracy = evaluations.count(True) / len(evaluations)

    return accuracy, weighted_f1, macro_f1, raw_predictions, c_pred, evaluations, conf_matrix

# ----- CoT generation and evaluation ----- #
def cot_generate_evaluate(model, dataloader, tokenizer, device, generation_kwargs):
    model.eval()
    raw_predictions_cot = []

    with torch.no_grad():
        for samples in tqdm(
            dataloader,
            total=len(dataloader),
            bar_format="{l_bar}{bar:25}{r_bar}",
            desc="CoT generation and evaluation",
            ascii=True,
        ):
            ### Generate CoT ###
            input_ids_c = samples["input_ids_c"].to(device)
            generate_inputs_c = {
                "input_ids": samples["input_ids_c"].to(device),
                "attention_mask": samples["attention_mask_c"].to(device),
                "pixel_values": samples["pixel_values"].to(device),
                "image_grid_thw": samples["image_grid_thw"].to(device)
            }
            generated_ids_c = model.generate(**generate_inputs_c, max_new_tokens=generation_kwargs["max_length"]).detach()
            generated_ids_trimmed_c = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_c, generated_ids_c)]

            output_texts_cot = tokenizer.batch_decode(
                generated_ids_trimmed_c, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw_predictions_cot.extend(output_texts_cot)

    gt_cots = [s["cot"] for s in dataloader.dataset]
    avg_bleu = calculate_pairwise_bleu_avg(raw_predictions_cot, gt_cots)
    avg_cos = sentence_transformers_similarity_batch(raw_predictions_cot, gt_cots)
    avg_meteor = calculate_pairwise_meteor_avg(raw_predictions_cot, gt_cots)
    avg_rouge_l = calculate_pairwise_rouge_l_avg(raw_predictions_cot, gt_cots)
    distinct_1, distinct_2 = calculate_distinct_1_2(gt_cots)

    return avg_bleu, avg_cos, avg_meteor, avg_rouge_l, distinct_1, distinct_2, raw_predictions_cot

seed_everything(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ----- Print out configurations ----- #
print("#" * 10, "Fine-tune-CoT Eval", "#" * 10)
print("\n".join(f"{k.ljust(25)}:{v}" for k, v in vars(args).items()))

# ----- Configurate Model, Tokenizer -----#
best_model, tokenizer, processor = get_model_and_tokenizer(model_size="7B", device=device,)

model_params = f"logs/models/ftcot/{args.teacher_model}/{args.dataset_key}_lr{args.ft_lr}_seed{args.seed}.pt"
best_model.load_state_dict(torch.load(model_params))
best_model.to(device)

evaluator = Evaluator(args.dataset_key, task_type="ft_cot_token")

# ----- Load & Prepare Dataset ----- #
print("Start processing the dataset...")
data_dir = f"data/{args.dataset_key}"
train_data_path_eval = os.path.join(data_dir, f"original_data_{args.teacher_model}", "train.json")
val_data_path = os.path.join(data_dir, f"original_data_{args.teacher_model}", "val.json")
test_data_path = os.path.join(data_dir, f"original_data_{args.teacher_model}", "test.json")

with open(train_data_path_eval) as f_train_eval ,open(val_data_path) as f_val, open(test_data_path) as f_test:
    train_json_data_eval = json.load(f_train_eval)
    val_json_data = json.load(f_val)
    test_json_data = json.load(f_test)

train_dataset_eval = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="test",
    data=train_json_data_eval,
    tokenizer=tokenizer,
    processor=processor
)

val_dataset = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="test",
    data=val_json_data,
    tokenizer=tokenizer,
    processor=processor
)

test_dataset = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="test",
    data=test_json_data,
    tokenizer=tokenizer,
    processor=processor
)

train_dataloader_eval = DataLoader(
    train_dataset_eval,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
)

del train_json_data_eval
del val_json_data
del test_json_data

print("Dataset processing completed.")

##### Evaluation #####
generation_kwargs = {"max_length": args.generate_max_length}

train_accuracy, train_f1, train_macro_f1, train_raw_predictions, train_c_pred, train_evaluations, train_cm = evaluate(
    best_model, train_dataloader_eval, tokenizer, evaluator, device, generation_kwargs
)
print(
    f"{args.dataset_key} || TRAIN Accuracy: {train_accuracy} Weighted_F1: {train_f1} Macro_F1: {train_macro_f1}"
)
print(train_cm)

val_accuracy, val_f1, val_macro_f1, val_raw_predictions, val_c_pred, val_evaluations, val_cm = evaluate(
    best_model, val_dataloader, tokenizer, evaluator, device, generation_kwargs
)
print(
    f"{args.dataset_key} || VAL Accuracy: {val_accuracy} Weighted_F1: {val_f1} Macro_F1: {val_macro_f1}"
)
print(val_cm)

test_accuracy, test_f1, test_macro_f1, test_raw_predictions, test_c_pred, test_evaluations, test_cm = evaluate(
    best_model, test_dataloader, tokenizer, evaluator, device, generation_kwargs
)

print(
    f"{args.dataset_key} || TEST Accuracy: {test_accuracy} Weighted_F1: {test_f1} Macro_F1: {test_macro_f1}"
)
print(test_cm)

test_avg_bleu, test_avg_cos, test_avg_meteor, test_avg_rouge_l, test_distinct_1, test_distinct_2, test_raw_predictions_cot = cot_generate_evaluate(best_model, test_dataloader, tokenizer, device, generation_kwargs)

print(f"Test Avg_BLEU: {test_avg_bleu}, Avg_Cosine_Similarity: {test_avg_cos}, Avg_Meteor: {test_avg_meteor}, Avg_Rouge_L: {test_avg_rouge_l}, Distinct_1: {test_distinct_1}, Distinct_2: {test_distinct_2}")