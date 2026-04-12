import os
import random
import pandas as pd

from clinicnumrobbench.data_creation.utils import format_df, RANDOM_STATE


comparison_type = "comparison"

def create_comparative_data(df, df_verbal, template_df, sub_out_dir):
    ##### first exceeded/less than/greater than
    comparison_comparative_type = f"{comparison_type}_comparative"
    question_template = "What is the record datetime when {vital_parameter} first {comparator} {threshold}?"
    vital_greater_than_threshold_mapping = {
        "sbp":120,
        "dbp":80,
        "heartrate":100,
        "resprate":20,
        "temperature":37.5,
    }
    vital_less_than_threshold_mapping = {
        "sbp":90,
        "dbp":60,
        "heartrate":60,
        "resprate":12,
        "temperature":36.5,
        "o2sat":95,
    }

    comparison_comparative_df = template_df.copy()
    vitals_counts = []

    # greater than
    comparison_comparative_exceeded_type = f"{comparison_comparative_type}_exceeded"
    for patient in df[df['vitals_count']>3].groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(4, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
        for vital_parameter in vital_greater_than_threshold_mapping:
            question = question_template.format(vital_parameter=vital_parameter, comparator=random.choice(["exceeded", "higher than"]), threshold=vital_greater_than_threshold_mapping[vital_parameter])
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            answer_record = patient_records[patient_records[vital_parameter] > vital_greater_than_threshold_mapping[vital_parameter]]['charttime']
            if len(answer_record) > 0:
                answer = answer_record.values[0]
                comparison_comparative_df.loc[len(comparison_comparative_df)] = {
                    "subject_id":patient,
                    "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                    "question":question, 
                    "answer":answer, 
                    "answer_index":answer_record.index.values[0],
                    "type":comparison_type, 
                    "sub_type":comparison_comparative_exceeded_type}
                vitals_counts.append(patient_records['vitals_count'].values[0])

    # less than
    comparison_comparative_drop_type = f"{comparison_comparative_type}_drop_below"
    for patient in df[df['vitals_count']>3].groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(4, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
        for vital_parameter in vital_less_than_threshold_mapping:
            question = question_template.format(vital_parameter=vital_parameter, comparator=random.choice(["less than", "drop below"]), threshold=vital_less_than_threshold_mapping[vital_parameter])
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            answer_record = patient_records[patient_records[vital_parameter] < vital_less_than_threshold_mapping[vital_parameter]]['charttime']
            if len(answer_record) > 0:
                answer = answer_record.values[0]
                comparison_comparative_df.loc[len(comparison_comparative_df)] = {
                        "subject_id":patient, 
                    "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                    "question":question, 
                    "answer":answer, 
                    "answer_index":answer_record.index.values[0],
                    "type":comparison_type, 
                    "sub_type":comparison_comparative_drop_type}
                vitals_counts.append(patient_records['vitals_count'].values[0])

    sub_comparison_type = "comparative"
    print(f"Distribution of {sub_comparison_type}: ", comparison_comparative_df['sub_type'].value_counts())
    # convert to final format
    formatted_comparison_comparative_df = format_df(comparison_comparative_df)
    # save
    formatted_comparison_comparative_df.to_csv(os.path.join(sub_out_dir, f"{comparison_comparative_type}.csv"), index=False)
    print(f"Saved {comparison_comparative_type} data with {len(formatted_comparison_comparative_df)} samples")
    

def create_superlative_data(df, df_verbal, template_df, sub_out_dir):
    comparison_superlative_type = f"{comparison_type}_superlative"

    question_template = "What is the record datetime when {vital_parameter} is {comparator}?"
    vital_parameters = ["temperature", "heartrate", "sbp", "dbp", "resprate", "o2sat"]
    comparators = ["highest", "lowest"]

    comparison_superlative_df = template_df.copy()
    vitals_counts = []
    # greater than
    for patient in df[df['vitals_count']>6].groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(1, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
        for vital_parameter in vital_parameters:
            for comparator in comparators:
                comparison_superlative_comparator_type = f"{comparison_superlative_type}_{comparator}"
                question = question_template.format(vital_parameter=vital_parameter, comparator=comparator)
                patient_record = df[df['subject_id']==patient].reset_index(drop=True).sort_values(by=vital_parameter, ascending=comparator == "lowest")
                answer = patient_record["charttime"].values[0]
                comparison_superlative_df.loc[len(comparison_superlative_df)] = {
                    "subject_id":patient, 
                    "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                    "question":question, 
                    "answer":answer, 
                    "answer_index":patient_record.iloc[0].name,
                    "type":comparison_type, 
                    "sub_type":comparison_superlative_comparator_type}
                vitals_counts.append(patient_record['vitals_count'].values[0])

    
    sub_comparison_type = "superlative"
    print(f"Distribution of {sub_comparison_type}: ", comparison_superlative_df['sub_type'].value_counts())
    # convert to final format
    formatted_comparison_superlative_df = format_df(comparison_superlative_df)
    # save
    formatted_comparison_superlative_df.to_csv(os.path.join(sub_out_dir, f"{comparison_superlative_type}.csv"), index=False)
    print(f"Saved {comparison_superlative_type} data with {len(formatted_comparison_superlative_df)} samples")


def create_comparison_data(df, df_verbal, template_df, output_dir):
    """
    Create comparison data from df and df_verbal
    1. Comparative
    2. Superlative

    Args:
        df: DataFrame with vitalsigns records, to get value for determistic answer programmatically
        df_verbal: DataFrame with chronological verbalized vitalsigns, to get context and question
        template_df: DataFrame to store the template data
        output_dir: Directory to store the final data

    Return:
        None with csv files saved in output_dir with format from format_df function
    """
    
    sub_out_dir = os.path.join(output_dir, comparison_type)
    os.makedirs(sub_out_dir, exist_ok=True)

    create_comparative_data(df, df_verbal, template_df, sub_out_dir)
    create_superlative_data(df, df_verbal, template_df, sub_out_dir)
