import os

from clinicnumrobbench.data_creation.utils import format_df, RANDOM_STATE


def create_retrieval_data(df, df_verbal, template_df, output_dir):
    """
    Create retrieval data from df and df_verbal
    We have 6 number columns: emperature,heartrate,resprate,o2sat,sbp,dbp 
    So, I will get from record 1 to 5 of 2 patients (1 male and 1 female)

    Args:
        df: DataFrame with vitalsigns records, to get value for determistic answer programmatically
        df_verbal: DataFrame with chronological verbalized vitalsigns, to get context and question
        template_df: DataFrame to store the template data
        output_dir: Directory to store the final data

    Return:
        None with csv files saved in output_dir with format from format_df function
    """

    retrieval_type = "retrieval"
    template1_sub_type = "direct_retrieval"

    output_dir = os.path.join(output_dir, retrieval_type)
    os.makedirs(output_dir, exist_ok=True)

    columns_verbal_mapping = {"temperature":"temperature", 
            "heartrate":"heart rate", 
            "resprate":"respiratory rate", 
            "o2sat":"O2 saturation", 
            "sbp":"systolic blood pressure", 
            "dbp":"diastolic blood pressure"}
    question_template = "What was the {column} of the patient at {charttime}?"

    num_records = [1,10]
    num_patients = 1
    template1_retrieval_df = template_df.copy()
    for num_record in num_records:
        # select patients with N records and 2 patients for each gender
        selected_subjects = df_verbal[df_verbal['vitals_count']==num_record].groupby(["gender", "age_cate"]).apply(lambda x: x.sample(n=num_patients, random_state=RANDOM_STATE))['subject_id'].tolist()
        for i in range(num_record):
            for patient in selected_subjects:
                for column in columns_verbal_mapping:
                    sample = df[df['subject_id']==patient].iloc[i]
                    question = question_template.format(column=columns_verbal_mapping[column], charttime=sample['charttime'])
                    answer = sample[column]
                    template1_retrieval_df.loc[len(template1_retrieval_df)] = {
                        "subject_id":patient, 
                        "context": df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                        "question":question, 
                        "answer":answer, 
                        "answer_index":i,
                        "type":retrieval_type, 
                        "sub_type":template1_sub_type}

    # convert to final format
    formatted_template1_retrieval_df = format_df(template1_retrieval_df)
    # save
    formatted_template1_retrieval_df.to_csv(os.path.join(output_dir, f"{template1_sub_type}.csv"), index=False)
    print(f"Saved {template1_sub_type} data with {len(formatted_template1_retrieval_df)} samples")
