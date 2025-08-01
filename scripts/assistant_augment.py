import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.getcwd())

import json
import torch
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from src.evaluator import Evaluator
from src.dataset import FinetuneCoTDataset
from src.functions import *
from src.process import filter_merge_sort_json


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["diverse_reasoning", "vanilla"], default="diverse_reasoning")
parser.add_argument("--teacher_model", type=str, default="gpt_4o_mini")
parser.add_argument("--dataset_key", type=str, default="SAS")
parser.add_argument("--n_diversity", type=int, default=1)
parser.add_argument("--model_max_length", type=int, default=1024)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--ft_cot_lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--generate_max_length", type=int, default=512)
args = parser.parse_args()

seed_everything(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Print out configurations ----- #
print("#" * 10, "Building Augmentation", "#" * 10)
print("\n".join(f"{k.ljust(25)}:{v}" for k, v in vars(args).items()))

# ----- Configurate Model, Tokenizer -----#
best_model, tokenizer, processor = get_model_and_tokenizer(model_size="7B", device=device)
evaluator = Evaluator(args.dataset_key, task_type="ft_cot_token")

model_params = f"logs/models/ftcot/{args.teacher_model}/{args.dataset_key}_lr{args.ft_cot_lr}_seed{args.seed}.pt"
best_model.load_state_dict(torch.load(model_params))
best_model.to(device)

# ---- Configurate Dataset ------ %
skeleton_data_path = f"data/{args.dataset_key}/original_data_{args.teacher_model}/train.json"  # use vanilla data, regardless of mode
with open(skeleton_data_path) as f:
    skeleton_data = json.load(f)

aug_dataset = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="test",
    data=skeleton_data,
    tokenizer=tokenizer,
    processor=processor
)

aug_dataloader = DataLoader(aug_dataset, batch_size=args.batch_size, num_workers=6, shuffle=False, pin_memory=True)

raw_predictions = []

with torch.no_grad():
    best_model = best_model.eval()
    generation_kwargs = {"max_length": args.generate_max_length}
    tqdm_format = tqdm(
        aug_dataloader,
        total=len(aug_dataloader),
        bar_format="{l_bar}{bar:25}{r_bar}",
        desc=f"{args.dataset_key}",
        ascii=True,
    )

    for sample in tqdm_format:
        # Prepare inputs
        input_ids = sample["input_ids_c"].to(device)
        generate_inputs = {
            "input_ids": sample["input_ids_c"].to(device),
            "attention_mask": sample["attention_mask_c"].to(device),
            "pixel_values": sample["pixel_values"].to(device),
            "image_grid_thw": sample["image_grid_thw"].to(device)
        }

        if args.mode == "vanilla":
            generated_ids = best_model.generate(**generate_inputs, max_new_tokens=generation_kwargs["max_length"]).detach()

        elif args.mode == "diverse_reasoning":
            generated_ids = best_model.generate(
                **generate_inputs, 
                max_new_tokens=generation_kwargs["max_length"],
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=args.n_diversity
            ).detach()
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
        output_texts_cot = tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        raw_predictions.extend(output_texts_cot)

raw_answers = [element for element in aug_dataset.raw_answers for _ in range(args.n_diversity)]
evaluations, c_preds, f1, macro_f1, conf_matrix = return_evaluations_in_boolean(
    evaluator, raw_predictions, raw_answers, return_cleansed_predictions=True
)
eval_correct = evaluations.count(True)
accuracy = eval_correct / len(evaluations)
raw_chains = raw_predictions

# ----- Store Augmented Data to Directory ----- #
# Prepare elements to store
inputs_id = [s["id"] for s in skeleton_data for _ in range(args.n_diversity)]
inputs_text = [s["text"] for s in skeleton_data for _ in range(args.n_diversity)]
inputs_image_path = [s["image_path"] for s in skeleton_data for _ in range(args.n_diversity)]
labels = [s["label"] for s in skeleton_data for _ in range(args.n_diversity)]

data_to_augment = []
for inpid, inpt, inpip, ch, comp, ans, v, s in zip(
    inputs_id,
    inputs_text,
    inputs_image_path,
    raw_chains,
    c_preds,
    labels,
    evaluations,
    skeleton_data * args.n_diversity,
):
    item = {
        "id": inpid,
        "text": inpt,
        "image_path": inpip,
        "cot": str(ch),
        "completion": str(comp),
        "label": ans,
        "initial_correct": v,
    }
    if args.dataset_key in ("T15", "T17"):
        item["aspect"] = s["aspect"]
    
    data_to_augment.append(item)

aug_dir = f"data/{args.dataset_key}"
if not os.path.exists(aug_dir):
    os.makedirs(aug_dir)

aug_path = os.path.join(aug_dir, f"aug{args.n_diversity}_{args.mode}_{args.teacher_model}.json")
with open(aug_path, "w") as f:
    json.dump(data_to_augment, f, indent=4)

# Finally, print out overall results
print(f"{args.dataset_key} | {args.teacher_model} || Data: {eval_correct}/{len(data_to_augment) * args.n_diversity} || Acc: {accuracy*100:.4f} || Weighted_F1: {f1:.4f} || Macro_F1: {macro_f1:.4f}")
print(conf_matrix)

peft_file = f"data/{args.dataset_key}/peft_{args.teacher_model}.json"
output_file = f"data/{args.dataset_key}/merge_{args.mode}_{args.teacher_model}.json"
filter_merge_sort_json(peft_file, aug_path, output_file)
