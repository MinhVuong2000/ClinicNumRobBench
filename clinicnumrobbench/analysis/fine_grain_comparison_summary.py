import os
import re
import argparse

import pandas as pd


def evaluate(data_file: str, task: str):
    """
    Evaluate the model on the data file.
    Args:
        data_file: The path to the data file.
    """
    df = pd.read_csv(data_file)

    number_long_extract_pattern = lambda x: x[x.find('"answer":')+10:-1]
    text_extract_pattern = lambda x: x[x.find('"answer":')+11:-2]

    pattern = r'-?\d*\.?\d+'

    if task == "comparison":
        df['llm_final_answer'] = df['generated_text'].map(text_extract_pattern)
        df['correct'] = df.apply(lambda r: r['llm_final_answer'].lower()==str(r['answer']).lower(), axis=1)
    elif task == "summary":
        df['llm_final_answer'] = df['generated_text'].map(number_long_extract_pattern)
        df['llm_final_answer'] = df['llm_final_answer'].apply(lambda x: x[re.search(pattern,x).span()[0]:re.search(pattern,x).span()[1]] if re.search(pattern,x) else "")
        df['correct'] = df.apply(lambda r: str(r['answer']).strip()==r['llm_final_answer'].strip() or re.sub(r'\.0+$', '', str(r['answer'])).strip()==re.sub(r'\.0+$', '', r['llm_final_answer'].strip()), axis=1)
    else:
        raise ValueError(f"Task {task} not supported")

    return df[['dataset_source', 'correct']].groupby('dataset_source')['correct'].mean().map(lambda x: round(x*100,2)).to_dict()


def main(data_dir: str, prompt_type: str, task: str):
    """
    Evaluate the model on the data file.
    Args:
        data_file: The path to the data file.
        problem: The problem to evaluate on.
        prompt_type: The type of prompt to use.
    """
    for lv in [task]:
        for context in ['structured', 'templated','natural']:
            results = {}
            context_dir = os.path.join(data_dir, context, lv, prompt_type)
            problems = os.listdir(context_dir)
            for problem in problems:
                problem_dir = os.path.join(context_dir, problem)
                for model_name_file in os.listdir(problem_dir):
                    data_path = os.path.join(problem_dir, model_name_file)
                    if not os.path.exists(data_path):
                        continue
                    try:
                        acc = evaluate(data_path, task=task)
                        if model_name_file.replace(".csv", "") not in results:
                            results[model_name_file.replace(".csv", "")] = acc # dict
                        results[model_name_file.replace(".csv", "")].update(acc)
                    except Exception as e:
                        print(f"Error evaluating {context} {lv} {prompt_type} {problem} {model_name_file}...")
                        print(f"Error details: {e}")
                        continue
            output_dir = os.path.join(data_dir, "evaluation", "fine_grain", prompt_type, task)
            os.makedirs(output_dir, exist_ok=True)
            out_df = pd.DataFrame(results).T
            out_df.columns = [c.replace(f"{task}_", "") for c in out_df.columns]
            out_df.to_markdown(os.path.join(output_dir,  f"{context}.md"))

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
        default="zero_shot_cot",
    )
    argparser.add_argument(
        "--task",
        type=str,
        default="comparison",
        choices=["comparison", "summary"],
    )
    args = argparser.parse_args()
    main(data_dir=args.data_dir, prompt_type=args.prompt_type, task=args.task)

# python3.11 clinicnumrobbench/analysis/fine_grain_comparison_summary.py --prompt_type zero_shot_cot --data_dir outputs
