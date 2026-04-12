import os
import json
from termcolor import cprint

from lexical_diversity import lex_div as ld
import pandas as pd
from datasets import load_dataset


def ld_per_question(question: str) -> float:
    """
    Calculate the lexical diversity of a question.
    """
    flt = ld.flemmatize(question)

    return {
        "hdd": ld.hdd(flt),
        "mtld": ld.mtld(flt),
        # "ttr": ld.ttr(flt),
        "log_ttr": ld.log_ttr(flt),
        "root_ttr": ld.root_ttr(flt),
        # "mtld_ma_wrap": ld.mtld_ma_wrap(flt),
        # "mtld_ma_bid": ld.mtld_ma_bid(flt),
    }


def check_lexical_diversity_per_df(df: pd.DataFrame, text_col: str = "question"):
    """
    Check the lexical diversity of the text column in the dataframe.
    """
    ld_df = pd.json_normalize(df[text_col].map(lambda x: ld_per_question(x)))
    return ld_df.mean().map(lambda x: round(x, 4)).to_dict()


def main():
    """
    Main function to check the lexical diversity of the data.
    """
    results = []
    for context in ["natural"]: # "structured", "templated",
        for task in ["retrieval", "calculation", "comparison", "summary"]:
            cprint(f"Checking lexical diversity of {context} {task}...", 'blue')
            full_task_df = pd.DataFrame()
            for subtask in os.listdir(os.path.join("data", context, task)):
                if full_task_df.empty:
                    full_task_df = pd.read_csv(os.path.join("data", context, task, subtask))
                else:
                    full_task_df = pd.concat([full_task_df, pd.read_csv(os.path.join("data", context, task, subtask))])
            ld_data = {"context": context, "task": task}
            ld_data.update(check_lexical_diversity_per_df(full_task_df))
            results.append(ld_data)

    report_df = pd.DataFrame(results)
    cprint(json.dumps(report_df[report_df['context'] == 'natural'].drop(columns=['context', 'task']).mean().to_dict(), indent=2), 'green')

def check_lexical_diversity_medcalc():
    df = pd.DataFrame(load_dataset("ncbi/MedCalc-Bench-v2.0", split="test"))
    df['question'] = df.apply(lambda x: f"{x['Patient Note']}\n\n{x['Question']}", axis=1)
    ld_data = check_lexical_diversity_per_df(df)
    cprint(json.dumps(ld_data, indent=2), 'green')

if __name__ == "__main__":
    cprint("Checking lexical diversity of MedNumBench...", 'blue')
    main()
    print()
    cprint("Checking lexical diversity of MedCalc-Bench...", 'blue')
    check_lexical_diversity_medcalc()
