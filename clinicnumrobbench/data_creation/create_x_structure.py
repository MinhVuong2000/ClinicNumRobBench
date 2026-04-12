import os
import re
import textwrap
import pandas as pd
from argparse import ArgumentParser

from clinicnumrobbench.data_creation.utils import format_df, RANDOM_STATE
from clinicnumrobbench.data_creation.create_retrieval_data import create_retrieval_data
from clinicnumrobbench.data_creation.create_arthmetic_data import create_arthmetic_data
from clinicnumrobbench.data_creation.create_comparison_data import create_comparison_data
from clinicnumrobbench.data_creation.create_aggregation_data import create_aggregation_data


def verbalize(df, context_model):
    if context_model == "templated":
        standard_template = "At {charttime}, the {age}-year-old {gender} reports pain level is {pain}. {possessive_pronoun} vital sign at the time is temperature {temperature}°C, heart rate {heartrate} bpm, respiratory rate {resprate} bpm, O2 saturation {o2sat}%, blood pressure {sbp}/{dbp} mmHg."
    elif context_model == "structured":
        standard_template = textwrap.dedent("""{{
        chart_time: {charttime},
        age: {age},
        gender: {gender},
        pain: {pain},
        temperature: {temperature},
        heart_rate: {heartrate},
        resp_rate: {resprate},
        o2_sat: {o2sat},
        bp: {sbp}/{dbp}
    }}""")
    else:
        raise ValueError(f"Invalid context model: {context_model}")

    return df.apply(lambda row: standard_template.format(**row), axis=1)


def group_vitalsigns_by_asc_time(df, verbal_col_name, context_model):
    if context_model == "templated":
        split_char = "\n"
    elif context_model == "structured":
        split_char = ",\n"
    else:
        raise ValueError(f"Invalid context model: {context_model}")
        
    # Group by subject_id and merge standard_verbal with "\n\n"
    # short by record time
    return split_char, df.sort_values(['subject_id', 'charttime']).reset_index(drop=True).groupby('subject_id')[verbal_col_name].apply(lambda x: split_char.join(x)).reset_index()


def replace_age_gender_with_pronouns(text, split_char):
    # Replace age-gender patterns with pronouns for the first occurrence, then use pronouns for subsequent occurrences
    lines = text.split(split_char)
    if len(lines) <= 1:
        return text
    
    # Keep the first line as is, replace in subsequent lines
    result_lines = [lines[0]]
    for line in lines[1:]:
        line = line.replace(r'the \d+-year-old female', 'she', 1) if 'female' in line else line
        line = line.replace(r'the \d+-year-old male', 'he', 1) if 'male' in line else line
        # Use regex for more precise replacement
        import re
        line = re.sub(r'the \d+-year-old female', 'she', line)
        line = re.sub(r'the \d+-year-old male', 'he', line)
        result_lines.append(line)
    
    return split_char.join(result_lines)
    

def main(context_model, input_path, output_dir):
    df = pd.read_csv(input_path)
    df['charttime'] = pd.to_datetime(df['charttime'])
    df.drop(columns=["stay_id"], inplace=True)

    # Create gender fullname
    df['gender'] = df['gender'].map(lambda x: "male" if x == "M" else "female") # because just have M and F in the data
    print("Head of input df:\n", df.head())

    # Create possessive pronoun
    df['possessive_pronoun'] = df['gender'].map(lambda x: "His" if x == "male" else "Her")
    
    # Create standard verbalization
    verbal_col_name = 'standard_verbal'
    df[verbal_col_name] = verbalize(df, context_model)

    # short by record time
    split_char, df_verbal = group_vitalsigns_by_asc_time(df, verbal_col_name, context_model)

    # Replace age-gender patterns with pronouns from second record of each subject
    df_verbal[verbal_col_name] = df_verbal[verbal_col_name].apply(lambda t: replace_age_gender_with_pronouns(t, split_char.replace(",","")))
    print("Sample: ",df_verbal[verbal_col_name][20])

    # Add vitals_count, gender, age columns to df_grouped based on subject_id
    # Get the first occurrence of each subject_id to extract these columns
    subject_info = df.groupby('subject_id')[['vitals_count', 'gender', 'age', 'age_cate']].first().reset_index()
    df_verbal = df_verbal.merge(subject_info, on='subject_id', how='left')

    template_df = pd.DataFrame(columns=["subject_id","context", "question", "answer", "answer_index","type", "sub_type"])
    create_retrieval_data(df, df_verbal, template_df, output_dir)
    create_arthmetic_data(df, df_verbal, template_df, output_dir)
    create_comparison_data(df, df_verbal, template_df, output_dir)
    create_aggregation_data(df, df_verbal, template_df, output_dir)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--context-mode", type=str, required=True, choices=["structured", "templated"])
    args.add_argument("--root-output-dir", type=str, default="data/")
    args.add_argument("--input-path", type=str, default="data/mimiciv/mimic4ed/200_sampled.csv")
    
    args = args.parse_args()
    
    output_dir = os.path.join(args.root_output_dir, args.context_mode)
    main(args.context_mode, args.input_path, output_dir)
