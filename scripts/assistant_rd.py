import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.getcwd())

import copy
import json
import argparse
import torch

from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm.auto import tqdm

from src.dataset import FinetuneCoTDataset
from src.functions import *
from src.evaluator import Evaluator

# ----- Main script ----- #
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_key", type=str, default="SAS")
parser.add_argument("--teacher_model", type=str, default="gpt_4o_mini")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--test_batch_size", type=int, default=24)
parser.add_argument("--model_max_length", type=int, default=1024)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--generate_max_length", type=int, default=512)
parser.add_argument("--lr_patience", type=int, default=2)
parser.add_argument("--lr_factor", type=float, default=0.5)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
args = parser.parse_args()

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, factor=args.lr_factor, min_lr=args.min_lr
    )

# ----- Function to evaluate on any dataset (val/test) ----- #
def evaluate(model, dataloader, tokenizer, evaluator, device, generation_kwargs):
    model.eval()
    raw_predictions_label = []

    with torch.no_grad():
        for samples in tqdm(
            dataloader,
            total=len(dataloader),
            bar_format="{l_bar}{bar:25}{r_bar}",
            desc="Evaluating",
            ascii=True,
        ):
            ### Predict the label ###
            input_ids_l = samples["input_ids_l"].to(device)
            generate_inputs_l = {
                "input_ids": samples["input_ids_l"].to(device),
                "attention_mask": samples["attention_mask_l"].to(device),
                "pixel_values": samples["pixel_values"].to(device),
                "image_grid_thw": samples["image_grid_thw"].to(device)
            }
            generated_ids_l = model.generate(**generate_inputs_l, max_new_tokens=generation_kwargs["max_length"]).detach()
            generated_ids_trimmed_l = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids_l, generated_ids_l)]

            output_texts_label = tokenizer.batch_decode(
                generated_ids_trimmed_l, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            raw_predictions_label.extend(output_texts_label)

    raw_answers = [s["label"] for s in dataloader.dataset]
    evaluations, c_pred, weighted_f1, macro_f1, conf_matrix = return_evaluations_in_boolean(
        evaluator, raw_predictions_label, raw_answers, return_cleansed_predictions=True
    )
    accuracy = evaluations.count(True) / len(evaluations)

    return accuracy, weighted_f1, macro_f1, raw_predictions_label, c_pred, evaluations, conf_matrix

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
print("#" * 10, "Stage1: Fine-tune-CoT Training", "#" * 10)
print("\n".join(f"{k.ljust(25)}:{v}" for k, v in vars(args).items()))

# ----- Configurate Model, Tokenizer -----#
model, tokenizer, processor = get_model_and_tokenizer(model_size="7B", device=device)
evaluator = Evaluator(args.dataset_key, task_type="ft_cot_token")

# ----- Load & Prepare Dataset ----- #
print("Start processing the dataset...")
data_dir = f"data/{args.dataset_key}"
train_data_path = os.path.join(data_dir, f"peft_{args.teacher_model}.json")
train_data_path_eval = os.path.join(data_dir, f"original_data_{args.teacher_model}", "train.json")
val_data_path = os.path.join(data_dir, f"original_data_{args.teacher_model}", "val.json")
test_data_path = os.path.join(data_dir, f"original_data_{args.teacher_model}", "test.json")

with open(train_data_path) as f_train, open(train_data_path_eval) as f_train_eval ,open(val_data_path) as f_val, open(
    test_data_path
) as f_test:
    train_json_data = json.load(f_train)
    train_json_data_eval = json.load(f_train_eval)
    val_json_data = json.load(f_val)
    test_json_data = json.load(f_test)

train_dataset = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="train",
    data=train_json_data,
    tokenizer=tokenizer,
    processor=processor
)

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

train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
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

del train_json_data
del train_json_data_eval
del val_json_data
del test_json_data

print("Dataset processing completed.")

# ----- Configure training-related elements ----- #
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = get_scheduler(optimizer, args)

# Save best model(Path)
model_save_dir = f"logs/models/ftcot/{args.teacher_model}"
os.makedirs(model_save_dir, exist_ok=True)
model_name = f"{args.dataset_key}_lr{args.lr}_seed{args.seed}.pt"
save_path = os.path.join(model_save_dir, model_name)

# ----- Train and Evaluate! ----- #
step = 0
best_accuracy = 0
final_f1 = 0
final_macro_f1 = 0
best_test_accuracy = 0
final_test_f1 = 0
final_test_macro_f1 = 0
best_epoch = 0
global_step = 0
is_best = 0
best_test_gen = []
best_test_gen_label = []
best_model_state_dict = None
for epoch in range(1, args.epoch + 1):
    model = model.train()
    optimizer.zero_grad()
    total_loss_c = 0.0
    total_loss_l = 0.0
    tqdm_format = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        bar_format="{l_bar}{bar:25}{r_bar}",
        desc=f"Epoch #{epoch}",
        ascii=True,
    )

    for train_data in tqdm_format:
        # CoT datas
        kwargs_c = {
            "input_ids": train_data["input_ids_c"],
            "attention_mask": train_data["attention_mask_c"],
            "labels": train_data["labels_c"],
            "pixel_values": train_data["pixel_values"],
            "image_grid_thw": train_data["image_grid_thw"]
        }
        kwargs_c = {k: v.to(device) for k, v in kwargs_c.items()}
        # Label datas
        kwargs_l = {
            "input_ids": train_data["input_ids_l"],
            "attention_mask": train_data["attention_mask_l"],
            "labels": train_data["labels_l"],
            "pixel_values": train_data["pixel_values"],
            "image_grid_thw": train_data["image_grid_thw"]
        }
        kwargs_l = {k: v.to(device) for k, v in kwargs_l.items()}


        outputs_c = model(**kwargs_c)
        loss_c = outputs_c["loss"]
        loss_c.backward()

        outputs_l = model(**kwargs_l)
        loss_l = outputs_l["loss"]
        loss_l.backward()

        # total_loss = 0.8 * loss_c + 0.2 * loss_l

        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss_c += loss_c.item()
        total_loss_l += loss_l.item()
        step += 1

    print(f"{args.dataset_key} || TRAIN Epoch #{epoch} Loss_CoT: {total_loss_c / len(train_dataloader):.4f} Loss_Label: {total_loss_l / len(train_dataloader):.4f}")

    ##### VAL Evaluation #####
    generation_kwargs = {"max_length": args.generate_max_length}
    
    if epoch == args.epoch:
        train_accuracy, train_f1, train_macro_f1, train_raw_predictions_label, train_c_pred, train_evaluations, train_cm = evaluate(
            model, train_dataloader_eval, tokenizer, evaluator, device, generation_kwargs
        )
        print(
            f"{args.dataset_key} || TRAIN Epoch #{epoch} Accuracy: {train_accuracy} Weighted_F1: {train_f1} Macro_F1: {train_macro_f1}"
        )
        print(train_cm)

    val_accuracy, val_f1, val_macro_f1, val_raw_predictions_label, val_c_pred, val_evaluations, val_cm = evaluate(
        model, val_dataloader, tokenizer, evaluator, device, generation_kwargs
    )
    
    if args.dataset_key in ("SAS", "SAM"):
        current_f1 = val_f1
        final_f1_metric = final_f1
    elif args.dataset_key in ("T15", "T17"):
        current_f1 = val_macro_f1
        final_f1_metric = final_macro_f1

    if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and current_f1 > final_f1_metric):
        is_best = 1
        best_accuracy = val_accuracy
        final_f1 = val_f1
        final_macro_f1 = val_macro_f1
        best_epoch = epoch
        # Save the current best model.
        best_model_state_dict = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    print(f"{args.dataset_key} || VAL Epoch #{epoch} Accuracy: {val_accuracy} Weighted_F1: {val_f1} Macro_F1: {val_macro_f1}|| Current Best: {best_accuracy} (Epoch #{best_epoch})")
    print(val_cm)

    scheduler.step(val_accuracy)

    test_accuracy, test_f1, test_macro_f1, test_raw_predictions_label, test_c_pred, test_evaluations, test_cm = evaluate(
        model, test_dataloader, tokenizer, evaluator, device, generation_kwargs
    )

    if is_best:
        is_best = 0
        best_test_accuracy = test_accuracy
        final_test_f1 = test_f1
        final_test_macro_f1 = test_macro_f1

        current_best_gen = []
        for i, data in enumerate(test_dataset):
            result_dict = {
                "label_prediction": test_c_pred[i],
                "is_correct": test_evaluations[i],
            }
            current_best_gen.append(result_dict)
        best_test_gen_label = current_best_gen

    print(f"{args.dataset_key} || TEST Epoch #{epoch} Accuracy: {test_accuracy} Weighted_F1: {test_f1} Macro_F1: {test_macro_f1} || Current Best: {best_test_accuracy} (Epoch #{best_epoch})")
    print(test_cm)

    # Reduce GPU fragmentations if any
    torch.cuda.empty_cache()

# ----- Post-training Procedures ----- #
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, save_path)
    print(f"Best model from epoch {best_epoch} saved to {save_path}")

best_model, tokenizer, processor = get_model_and_tokenizer(model_size="7B", device=device)
best_model.load_state_dict(torch.load(save_path))
model.to(device)

generation_dir = f"logs/gen_outputs/ftcot/{args.teacher_model}"
os.makedirs(generation_dir, exist_ok=True)
test_avg_bleu, test_avg_cos, test_avg_meteor, test_avg_rouge_l, test_distinct_1, test_distinct_2, test_raw_predictions_cot = cot_generate_evaluate(best_model, test_dataloader, tokenizer, device, generation_kwargs)
for i, data in enumerate(test_dataset):
    result_dict = {
        "id": data["id"],
        "label": data["label"],
        "text": data["text"],
        "image_path": data["image_path"],
        "generation_cot": test_raw_predictions_cot[i]
    }
    result_dict["label_prediction"] = best_test_gen_label[i]["label_prediction"]
    result_dict["is_correct"] = best_test_gen_label[i]["is_correct"]
    if args.dataset_key in ("T15", "T17"):
        result_dict["aspect"] = data["aspect"]
    best_test_gen.append(result_dict)

gen_file_name = f"{args.dataset_key}_seed{args.seed}_epoch{best_epoch}_test.json"
with open(os.path.join(generation_dir, gen_file_name), "w") as f:
    json.dump(best_test_gen, f, indent=4)

# Ultimately!
print(f"Saved best epoch model: Epoch #{best_epoch}, Val_Accuracy: {best_accuracy}, Val_F1: {final_f1}, Val_Macro_F1: {final_macro_f1}, Test_Accuracy: {best_test_accuracy}, Test_F1: {final_test_f1}, Test_Macro_F1: {final_test_macro_f1}")
print(f"Test Avg_BLEU: {test_avg_bleu}, Avg_Cosine_Similarity: {test_avg_cos}, Avg_Meteor: {test_avg_meteor}, Avg_Rouge_L: {test_avg_rouge_l}, Distinct_1: {test_distinct_1}, Distinct_2: {test_distinct_2}")