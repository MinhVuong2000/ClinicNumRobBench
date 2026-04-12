import os
import random
import pandas as pd

from clinicnumrobbench.data_creation.utils import format_df, RANDOM_STATE



def create_aggregation_data(df, df_verbal, template_df, output_dir):
    """
    Create aggregation data from df and df_verbal

    Args:
        df: DataFrame with vitalsigns records, to get value for determistic answer programmatically
        df_verbal: DataFrame with chronological verbalized vitalsigns, to get context and question
        template_df: DataFrame to store the template data
        output_dir: Directory to store the final data

    Return:
        None with csv files saved in output_dir with format from format_df function
    """

    summary_type = "summary"
    template1_sub_type = "summary"

    output_dir = os.path.join(output_dir, summary_type)
    os.makedirs(output_dir, exist_ok=True)

    question_template = "How many time the patient has {issue}? Return the number of records."
    issues = {
        "Tachycardia":{"heartrate":100}, 
        "Mean Arterial Pressure higher than 100 mmHg": {"map":100},
        "Shock Index higher than 0.7": {"shock_index":0.7}
    }

    summary_df = template_df.copy()
    vitals_counts = []
    df['map'] = round((df['sbp'] - df['dbp']) / 3 + df['dbp'], 1)
    df['shock_index'] = round(df['heartrate'] / df['sbp'], 1)
    for patient in df[df['vitals_count']>3].groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(50, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
        for issue in issues:
            summary_issue_type = f"summary_count_by_{issue.lower().replace(' ', '.')}"
            
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            filtered_records = patient_records[patient_records['heartrate'] > 100 if issue == "Tachycardia" else patient_records['map'] > 100 if issue == "Mean Arterial Pressure higher than 100 mmHg" else patient_records['shock_index'] > 0.7]
            question = question_template.format(issue=issue)
            answer = filtered_records.shape[0]
            summary_df.loc[len(summary_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":answer, 
                "answer_index":", ".join(filtered_records.index.values.astype(str)) if len(filtered_records) > 0 else '',
                "type":summary_type,    
                "sub_type":summary_issue_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])
                
    # convert to final format
    formatted_summary_df = format_df(summary_df)
    # save
    formatted_summary_df.to_csv(os.path.join(output_dir, f"{template1_sub_type}.csv"), index=False)
    print(f"Saved {template1_sub_type} data with {len(formatted_summary_df)} samples")
