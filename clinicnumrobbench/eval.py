import os
import re
import argparse

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from const import HF_TOKEN


def expected_calibration_error(y_true, y_pred_proba, n_bins=10):
    """
    ECE metric compatible with sklearn
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_pred_proba > bin_boundaries[i]) & (y_pred_proba <= bin_boundaries[i+1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            accuracy = np.mean(y_true[in_bin])
            confidence = np.mean(y_pred_proba[in_bin])
            ece += (bin_size / len(y_pred_proba)) * np.abs(accuracy - confidence)
    return ece


def get_answer_logprobs(tokenizer: AutoTokenizer, answer: str, logprobs: torch.Tensor, pattern: str = '"answer":'):
    """
    Get the answer probabilities for the data frame.
    Args:
        answer: The answer.
        logprobs: The logprobs.
    """
    cut_index = answer.find(pattern)+10
    cut_text = answer[:cut_index]
    answer_logprobs = logprobs[tokenizer(cut_text).input_ids.__len__():tokenizer(answer).input_ids.__len__()-1] # ignore the last token } and previous tokens as reasoning tokens
    return answer_logprobs


def evaluate(data_file: str, problem: str, tokenizer: AutoTokenizer):
    """
    Evaluate the model on the data file.
    Args:
        data_file: The path to the data file.
        problem: The problem to evaluate on.
    """
    df = pd.read_csv(data_file)
    number_long_extract_pattern = lambda x: x[x.find('"answer":')+10:-1] if x.find('"answer":') != -1 else x[x.find('""answer"":')+12:-1]
    text_extract_pattern = lambda x: x[x.find('"answer":')+11:-2] if x.find('"answer":') != -1 else x[x.find('""answer"":')+13:-2]

    pattern = r'-?\d*\.?\d+'

    if problem in ["comparative", "superlative"] or "ablation_var_comparison" in data_file:
        df['llm_final_answer'] = df['generated_text'].map(text_extract_pattern)
        df['correct'] = df.apply(lambda r: r['llm_final_answer'].lower()==str(r['answer']).lower() or re.sub(r'`|}|"', '', r['llm_final_answer']).lower()==str(r['answer']).lower(), axis=1)
    else:
        df['llm_final_answer'] = df['generated_text'].map(number_long_extract_pattern)
        df['llm_final_answer'] = df['llm_final_answer'].apply(lambda x: x[re.search(pattern,x).span()[0]:re.search(pattern,x).span()[1]] if re.search(pattern,x) else "")
        df['correct'] = df.apply(lambda r: str(r['answer']).strip()==r['llm_final_answer'].strip() or re.sub(r'\.0+$', '', str(r['answer'])).strip()==re.sub(r'\.0+$', '', r['llm_final_answer'].strip()), axis=1)

    acc = round(df['correct'].mean()*100, 2)

    if 'logprobs' in df.columns:
        df['logprobs_list'] = df['logprobs'].apply(lambda x: torch.tensor(eval(x.replace(', -inf', ', float("-inf")').replace(', inf', ', float("inf")'))) if isinstance(x, str) else x)
        df['conf'] = df['logprobs_list'].apply(lambda x: np.exp(x.numpy()).sum(axis=0)/x.shape[0])
        conf = round(df['conf'].mean()*100, 2)

        if 'ans_logprobs' not in df.columns:
            df['ans_logprobs'] = df.apply(lambda r: get_answer_logprobs(tokenizer, r['generated_text'], r['logprobs_list']), axis=1)
        df['ans_conf'] = df['ans_logprobs'].apply(lambda x: np.exp(x.numpy()).sum(axis=0)/x.shape[0])
        ans_conf = round(df['ans_conf'].mean()*100, 2)

        n_bins = 10
        ece = expected_calibration_error(df['correct'], df['conf'], n_bins=n_bins)
        ans_ece = expected_calibration_error(df['correct'], df['ans_conf'], n_bins=n_bins)
    else:
        conf, ece, ans_conf, ans_ece = 0, 0, 0, 0
    return acc, conf, ece, ans_conf, ans_ece


def main(data_dir: str, prompt_type: str):
    """
    Evaluate the model on the data file.
    Args:
        data_file: The path to the data file.
        problem: The problem to evaluate on.
        prompt_type: The type of prompt to use.
    """
    model_names_results = {}
    model_names_summary = {}
    for lv in ["lv1", "lv2", "lv3", "lv4", "lv4_full", "lv4_full_filtered", "retrieval", "calculation", "comparison", "summary"]:
        if lv == "lv4_full_filtered":
            lv = "lv4_full"
            lv4_filtered = True
        else:
            lv4_filtered = False
        try:
            problems = os.listdir(os.path.join(data_dir, lv, prompt_type))
        except FileNotFoundError:
            continue
        for problem in problems:
            if problem not in [
                "direct_retrieval",
                "calculation_1step", "calculation_2step", "calculation_3step",
                "comparative", "superlative",
                "summary", "abbr", "no_unit", "order","sep", "abbr_no_unit"
            ]:
                continue
            for model_name_file in os.listdir(os.path.join(data_dir, lv, prompt_type, problem)):
                model_name = model_name_file.replace(".csv", "")
                if "gpt-4.1" in model_name or "gpt-5" in model_name:
                    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", padding_side="left", token=HF_TOKEN or None)
                else:
                    try:
                        if "meta.llama3-3-70b-instruct-v1:0" in model_name: # bedrock model name has :0
                            hf_model_name = model_name.replace("meta.llama3-3-70b-instruct-v1:0","meta-llama/Llama-3.3-70B-Instruct")
                        else:
                            hf_model_name = model_name
                        tokenizer = AutoTokenizer.from_pretrained(hf_model_name.replace("__", "/").split("_0.")[0], padding_side="left", token=HF_TOKEN or None)
                    except OSError:
                        # bedrock use . instead of / in model name
                        hf_model_name = model_name.split(".")[0]+"/"+".".join(model_name.split(".")[1:])
                        tokenizer = AutoTokenizer.from_pretrained(hf_model_name.replace("__", "/").split("_0.")[0], padding_side="left", token=HF_TOKEN or None)
                    except Exception as e:
                        raise e
                data_path = os.path.join(data_dir, lv, prompt_type, problem, model_name_file)
                if not os.path.exists(data_path):
                    continue
                try:
                    acc, conf, ece, ans_conf, ans_ece = evaluate(data_path, problem, tokenizer)
                    if model_name not in model_names_results:
                        model_names_results[model_name] = []
                        model_names_summary[model_name] = {}

                    if lv not in model_names_summary[model_name]:
                        model_names_summary[model_name][lv] = {'len':[], 'acc':[], 'conf':[], 'ece':[], 'ans_conf':[], 'ans_ece':[]}
                    model_names_summary[model_name][lv]['len'].append(len(pd.read_csv(data_path)))
                    model_names_summary[model_name][lv]['acc'].append(acc)
                    model_names_summary[model_name][lv]['conf'].append(conf)
                    model_names_summary[model_name][lv]['ece'].append(ece)
                    model_names_summary[model_name][lv]['ans_conf'].append(ans_conf)
                    model_names_summary[model_name][lv]['ans_ece'].append(ans_ece)

                    model_names_results[model_name].append({
                        "lv": lv,
                        "problem": problem if not lv4_filtered else problem + "_filtered",
                        "acc": acc,
                        "conf": round(conf, 3),
                        "ece": round(ece, 3),
                        "ans_conf": round(ans_conf, 3),
                        "ans_ece": round(ans_ece, 3),
                    })
                except Exception as e:
                    print(f"Error evaluating {data_path} for {model_name}: {e}")
                    continue
    lambda_average_lv = lambda data, metric, decimal=2: round(sum(w * v for w, v in zip(data['len'], data[metric])) / sum(data['len']), decimal)
    for model_name, summary in model_names_summary.items():
        for lv, data in summary.items():
            model_names_results[model_name].append({
                "lv": lv,
                "problem": "avg",
                "acc": lambda_average_lv(data, 'acc', 2),
                "conf": lambda_average_lv(data, 'conf', 3),
                "ece": lambda_average_lv(data, 'ece', 3),
                "ans_conf": lambda_average_lv(data, 'ans_conf', 3),
                "ans_ece": lambda_average_lv(data, 'ans_ece', 3),
            })
    # lambda_avg = lambda summary, metric, round_decimal=3: round(sum([sum(w * v for w, v in zip(data['len'], data[metric])) for data in summary.values()]) / sum([sum(data['len']) for data in summary.values()]), round_decimal)
    # for model_name, summary in model_names_summary.items():
    #     model_names_results[model_name].append({
    #         "lv": "avg",
    #         "problem": "avg",
    #         "acc": lambda_avg(summary, 'acc',2),
    #         "conf": lambda_avg(summary, 'conf', 3),
    #         "ece": lambda_avg(summary, 'ece', 3),
    #         "ans_conf": lambda_avg(summary, 'ans_conf', 3),
    #         "ans_ece": lambda_avg(summary, 'ans_ece', 3),
    #     })
    for model_name, results in model_names_results.items():
        output_dir = os.path.join(data_dir, "evaluation", model_name)
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(results).to_markdown(os.path.join(output_dir, f"{prompt_type}.md"))
        print(f"Saved results for {model_name} to {output_dir}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_dir",
        type=str,
        default="outputs",
    )
    argparser.add_argument(
        "--prompt_type",
        type=str,
        default="zero_shot_direct_answer",
    )
    args = argparser.parse_args()
    main(data_dir=args.data_dir, prompt_type=args.prompt_type)

# python3.11 clinicnumrobbench/eval.py --data_dir outputs --prompt_type zero_shot_cot
