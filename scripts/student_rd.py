import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.getcwd())

import json
import warnings
import argparse
import copy

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import FinetuneCoTDataset
from src.kd_tools import VanillaKDLoss
from src.functions import *
from src.evaluator import Evaluator
import torch.optim as optim

warnings.filterwarnings("ignore")

# Output-layer soft label distillation
def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)

    loss = -targets_prob * student_likelihood  # .mean()
    loss = torch.sum(loss, dim=-1)

    return loss.mean()

# Hidden layer distillation
def hidden_distillation(teacher_reps, student_reps, linear_layer, kwargs):

    loss_mse = torch.nn.MSELoss()
    layers_per_block = int((len(teacher_reps) - 1) / (len(student_reps) - 1))
    student_layer_num = len(student_reps) - 1

    new_teacher_reps = [
        teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)
    ]
    new_student_reps = student_reps

    rep_loss = 0.0
    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        
        student_rep = student_rep[kwargs["labels"] != 0]
        teacher_rep = teacher_rep[kwargs["labels"] != 0]

        rep_loss += loss_mse(student_rep, linear_layer(teacher_rep))

    return rep_loss

# Attention distillation
def att_distillation(teacher_atts, student_atts):

    loss_mse = torch.nn.MSELoss()

    layers_per_block = int(len(teacher_atts) / len(student_atts))
    student_layer_num = len(student_atts)

    new_teacher_atts = [
        teacher_atts[i * layers_per_block + layers_per_block - 1]
        for i in range(student_layer_num)
    ]

    att_loss = 0.0
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        student_att = torch.where(
            student_att <= -1e2, torch.zeros_like(student_att).to(device), student_att
        )
        teacher_att = torch.where(
            teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device), teacher_att
        )
        att_loss += loss_mse(student_att, teacher_att)

    return att_loss

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_key", type=str, default="SAS")
parser.add_argument(
    "--n_aug_diversity",
    type=int,
    default=1,
    help="How many rationales were augmented by the mentor per question?",
)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--teacher_model", type=str, default="gpt_4o_mini")
parser.add_argument("--test_batch_size", type=int, default=24)
parser.add_argument("--teacher_model_size", type=str, default="7B")
parser.add_argument("--student_model_size", type=str, default="3B")
parser.add_argument("--model_max_length", type=int, default=1024)
parser.add_argument("--kd_temperature", type=float, default=1.0)
parser.add_argument("--kd_lambda", type=float, default=0.3)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--ft_cot_lr", type=float, default=3e-4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--generate_max_length", type=int, default=512)
parser.add_argument("--lr_patience", type=int, default=2)
parser.add_argument("--lr_factor", type=float, default=0.5)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--gradient_accumulation_steps", type=int, default=20)
parser.add_argument(
    "--training_mode",
    choices=["vanilla", "hidden", "mse", "ce", "none"],
    default="vanilla",
)
args = parser.parse_args()

seed_everything(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
teacher_device = torch.device("cuda:0")  # Assistant model
student_device = torch.device("cuda:1")  # Student model

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

def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, factor=args.lr_factor, min_lr=args.min_lr
    )

# Save best model(Path)
model_save_dir = f"logs/models/kd/{args.teacher_model}"
os.makedirs(model_save_dir, exist_ok=True)
model_name = f"{args.dataset_key}_lr{args.lr}_seed{args.seed}.pt"
save_path = os.path.join(model_save_dir, model_name)

def train_and_validate(
    args,
    teacher_model,
    student_model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    evaluator,
    device1,
    device2
):
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    
    scheduler = get_scheduler(optimizer, args)

    kd_criterion = VanillaKDLoss(temperature=args.kd_temperature)
    
    best_accuracy = 0
    best_f1 = 0
    best_macro_f1 = 0
    best_test_accuracy = 0
    best_test_f1 = 0
    best_test_macro_f1 = 0
    best_epoch = 0
    best_test_gen_label = []
    is_best = 0
    global_step = 0
    best_model_state_dict = None

    for epoch in range(1, args.epoch + 1):
        student_model.train()
        optimizer.zero_grad()
        total_loss_c_h = 0.0
        total_loss_c_s = 0.0
        total_loss_l_h = 0.0
        total_loss_l_s = 0.0
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
            kwargs_c_teacher = {k: v.to(device1) for k, v in kwargs_c.items()}
            kwargs_c_student = {k: v.to(device2) for k, v in kwargs_c.items()}

            # Respectively obtain the CoT soft logits of the assistant and the student.
            with torch.no_grad():
                teacher_output = teacher_model(**kwargs_c_teacher)
                teacher_logits = teacher_output["logits"][kwargs_c_teacher["labels"] != 0].to(device2)
                del teacher_output

            student_outputs = student_model(**kwargs_c_student)
            student_logits = student_outputs["logits"][kwargs_c_student["labels"] != 0]

            # Obtain the hard label CoT loss of the student.
            hard_rea_loss = student_outputs["loss"]
            soft_rea_loss = kd_criterion(student_logits, teacher_logits)

            cot_loss = 0.0
            if args.training_mode == "vanilla":
                cot_loss = ((1 - args.kd_lambda) * hard_rea_loss) + (args.kd_lambda * soft_rea_loss)
            else:
                cot_loss = hard_rea_loss
            cot_loss.backward()

            # Label datas
            kwargs_l = {
                "input_ids": train_data["input_ids_l"],
                "attention_mask": train_data["attention_mask_l"],
                "labels": train_data["labels_l"],
                "pixel_values": train_data["pixel_values"],
                "image_grid_thw": train_data["image_grid_thw"]
            }
            kwargs_l_teacher = {k: v.to(device1) for k, v in kwargs_l.items()}
            kwargs_l_student = {k: v.to(device2) for k, v in kwargs_l.items()}

            with torch.no_grad():
                teacher_output_l = teacher_model(**kwargs_l_teacher)
                teacher_logits_l = teacher_output_l["logits"][kwargs_l_teacher["labels"] != 0].to(device2)
                del teacher_output_l

            student_outputs_l = student_model(**kwargs_l_student)
            student_logits_l = student_outputs_l["logits"][kwargs_l_student["labels"] != 0]

            hard_cls_loss = student_outputs_l["loss"]
            soft_cls_loss = kd_criterion(student_logits_l, teacher_logits_l)

            cls_loss = 0.0
            if args.training_mode == "vanilla":
                cls_loss = ((1 - args.kd_lambda) * hard_cls_loss) + (args.kd_lambda * soft_cls_loss)
            else:
                cls_loss = hard_cls_loss
            cls_loss.backward()

            # total_loss = 0.8 * cot_loss + 0.2 * cls_loss

            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                # torch.cuda.empty_cache()

            total_loss_c_h += hard_rea_loss.item()
            total_loss_l_h += hard_cls_loss.item()
            total_loss_c_s += soft_rea_loss.item()
            total_loss_l_s += soft_cls_loss.item()

        print(f"{args.dataset_key} || TRAIN Epoch #{epoch} CoT_loss:{cot_loss / len(train_dataloader):.4f}, Cls_loss:{cls_loss / len(train_dataloader):.4f}, Hard_CoT_Loss: {total_loss_c_h / len(train_dataloader):.4f}, Soft_CoT_Loss: {total_loss_c_s / len(train_dataloader):.4f}, Hard_Label_Loss: {total_loss_l_h / len(train_dataloader):.4f}, Soft_Label_Loss: {total_loss_l_s / len(train_dataloader):.4f}")

        # Validation
        generation_kwargs = {"max_length": args.generate_max_length}

        if epoch == args.epoch:
            train_accuracy, train_f1, train_macro_f1, train_raw_predictions_label, train_c_pred, train_evaluations, train_cm = evaluate(
                student_model, train_dataloader_eval, student_tokenizer, evaluator, device2, generation_kwargs
            )
            print(
                f"{args.dataset_key} || TRAIN Epoch #{epoch} Accuracy: {train_accuracy} Weighted_F1: {train_f1} Macro_F1: {train_macro_f1}"
            )
            print(train_cm)

        val_accuracy, val_f1, val_macro_f1, val_raw_predictions_label, val_c_pred, val_evaluations, val_cm = evaluate(
            student_model, val_dataloader, student_tokenizer, evaluator, device2, generation_kwargs
        )

        if args.dataset_key in ("SAS", "SAM"):
            current_f1 = val_f1
            final_f1_metric = best_f1
        elif args.dataset_key in ("T15", "T17"):
            current_f1 = val_macro_f1
            final_f1_metric = best_macro_f1

        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and current_f1 > final_f1_metric):
            is_best = 1
            best_accuracy = val_accuracy
            best_f1 = val_f1
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            # Save the current best model.
            best_model_state_dict = copy.deepcopy({k: v.cpu() for k, v in student_model.state_dict().items()})

        print(f"{args.dataset_key} || VAL Epoch #{epoch} Accuracy: {val_accuracy} Weighted_F1: {val_f1} Macro_F1: {val_macro_f1}|| Current Best: {best_accuracy} (Epoch #{best_epoch})")
        print(val_cm)

        scheduler.step(val_accuracy)

        test_accuracy, test_f1, test_macro_f1, test_raw_predictions_label, test_c_pred, test_evaluations, test_cm = evaluate(
            student_model, test_dataloader, student_tokenizer, evaluator, device2, generation_kwargs
        )

        if is_best:
            is_best = 0
            best_test_accuracy = test_accuracy
            best_test_f1 = test_f1
            best_test_macro_f1 = test_macro_f1

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

    print(
        f"Saved best epoch model: Epoch #{best_epoch}, Val_Accuracy: {best_accuracy}, Val_F1: {best_f1}, Val_Macro_F1: {best_macro_f1}, Test_Accuracy: {best_test_accuracy}, Test_F1: {best_test_f1} Test_Macro_F1: {best_test_macro_f1}"
    )

    return best_epoch, best_accuracy, best_f1, best_macro_f1, best_test_gen_label, best_test_accuracy, best_test_f1, best_test_macro_f1, best_model_state_dict


# ----- Print out configurations ----- #
print("#" * 10, "MulCoT-RD Reasoning Distillation", "#" * 10)
print("\n".join(f"{k.ljust(25)}:{v}" for k, v in vars(args).items()))

# ----- Configurate Teacher & Student Model, Tokenizer -----#
teacher_model, teacher_tokenizer, teacher_processor = get_model_and_tokenizer(model_size=args.teacher_model_size, device=teacher_device)
teacher_model_path = f"logs/models/ftcot/{args.teacher_model}/{args.dataset_key}_lr{args.ft_cot_lr}_seed{args.seed}.pt"

teacher_model_params = torch.load(teacher_model_path)
teacher_model.load_state_dict(teacher_model_params)

for p in teacher_model.parameters():
    p.requires_grad = False

# Print prompts and memory usage information.
allocated_mem1 = torch.cuda.memory_allocated(teacher_device) / 1024**3
reserved_mem1 = torch.cuda.memory_reserved(teacher_device) / 1024**3
print(f"✅ 7B teacher model successfully loaded on device {teacher_device}! Current GPU memory usage:")
print(f"   Allocated memory: {allocated_mem1:.2f} GB")
print(f"   Reserved memory: {reserved_mem1:.2f} GB")

student_model, student_tokenizer, student_processor = get_model_and_tokenizer(model_size=args.student_model_size, device=student_device)
teacher_embedding_size = teacher_model.get_input_embeddings().weight.size(0)
# Use the teacher model's tokenizer consistently.
student_tokenizer = teacher_tokenizer
# Resize the student model's token embedding layer to match the vocabulary size of the teacher model's tokenizer.
student_model.resize_token_embeddings(teacher_embedding_size)

# Print prompts and memory usage information.
allocated_mem2 = torch.cuda.memory_allocated(student_device) / 1024**3
reserved_mem2 = torch.cuda.memory_reserved(student_device) / 1024**3
print(f"✅ 3B student model successfully loaded on device {student_device}! Current GPU memory usage:")
print(f"   Allocated memory: {allocated_mem2:.2f} GB")
print(f"   Reserved memory: {reserved_mem2:.2f} GB")

# ----- Load & Prepare Dataset ----- #
train_data_path = f"data/{args.dataset_key}/merge_diverse_reasoning_{args.teacher_model}.json"
val_data_path = f"data/{args.dataset_key}/original_data_{args.teacher_model}/val.json"
test_data_path = f"data/{args.dataset_key}/original_data_{args.teacher_model}/test.json"
with open(train_data_path) as f_train, open(val_data_path) as f_val, open(
    test_data_path
) as f_test:
    train_json_data = json.load(f_train)
    val_json_data = json.load(f_val)
    test_json_data = json.load(f_test)

train_dataset = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="train",
    data=train_json_data,
    tokenizer=student_tokenizer,
    processor=student_processor
)

train_dataset_eval = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="test",
    data=train_json_data,
    tokenizer=student_tokenizer,
    processor=student_processor
)

val_dataset = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="test",
    data=val_json_data,
    tokenizer=student_tokenizer,
    processor=student_processor
)

test_dataset = FinetuneCoTDataset(
    dataset_key=args.dataset_key,
    dataset_type="test",
    data=test_json_data,
    tokenizer=student_tokenizer,
    processor=student_processor
)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
train_dataloader_eval = DataLoader(train_dataset_eval, batch_size=args.test_batch_size, shuffle=False, num_workers=6, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=6, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=6, pin_memory=True)

evaluator = Evaluator(args.dataset_key, task_type="ft_cot_token")

# Delete unnecessary elements
del train_json_data, train_dataset_eval, val_json_data, test_json_data

best_student_model_state_dict=None

# ----- Training and Validation ----- #
best_epoch, best_accuracy, best_f1, best_macro_f1, best_test_gen_label, best_test_accuracy, best_test_f1, best_test_macro_f1, best_student_model_state_dict = train_and_validate(
    args,
    teacher_model,
    student_model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    evaluator,
    teacher_device,
    student_device
)

# ---- Saving the best epoch's generation results ----- #
if best_student_model_state_dict is not None:
    torch.save(best_student_model_state_dict, save_path)
    print(f"Best model from epoch {best_epoch} saved to {save_path}")

best_model, tokenizer, processor = get_model_and_tokenizer(model_size="3B", device=student_device)
tokenizer = teacher_tokenizer
best_model.resize_token_embeddings(teacher_embedding_size)
best_model.load_state_dict(torch.load(save_path))
best_model.to(student_device)

generation_kwargs = {"max_length": args.generate_max_length}
test_avg_bleu, test_avg_cos, test_avg_meteor, test_avg_rouge_l, test_distinct_1, test_distinct_2, test_raw_predictions_cot = cot_generate_evaluate(best_model, test_dataloader, tokenizer, student_device, generation_kwargs)
best_test_gen = []
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

gen_save_dir = f"logs/gen_outputs/kd/{args.teacher_model}"
os.makedirs(gen_save_dir, exist_ok=True)
gen_save_name = f"{args.dataset_key}_aug{args.n_aug_diversity}_seed{args.seed}_epoch{best_epoch}_test.json"
with open(os.path.join(gen_save_dir, gen_save_name), "w") as f:
    json.dump(best_test_gen, f, indent=4)

print(f"Test Avg_BLEU: {test_avg_bleu}, Avg_Cosine_Similarity: {test_avg_cos}, Avg_Meteor: {test_avg_meteor}, Avg_Rouge_L: {test_avg_rouge_l}, Distinct_1: {test_distinct_1}, Distinct_2: {test_distinct_2}")