import os
import pandas as pd

from clinicnumrobbench.data_creation.utils import format_df, RANDOM_STATE


calculation_type = "calculation"

def create_1step_data(df, df_verbal, template_df, sub_out_dir):
    def create_addition():
        # Addition - Sum of vital parameters
        question_template = "Calculate the sum of systolic and diastolic blood pressure at {charttime}?"
        addition_sub_type = "calc_1step_addition"

        calc_1step_addition_df = template_df.copy()
        # get 50 patients with 5 records each group by vitals_count
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records = patient_records.loc[len(patient_records)//3:].sample(n=1)
            question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
            answer = patient_records['sbp'].values[0] + patient_records['dbp'].values[0]
            calc_1step_addition_df.loc[len(calc_1step_addition_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":answer, 
                "answer_index":patient_records.index.values[0],
                "type":calculation_type, 
                "sub_type":addition_sub_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])

        print("Example question of {subtraction_sub_type}: ", calc_1step_addition_df['question'].values[0])
        return calc_1step_addition_df
    
    def create_subtraction():
        # Subtraction: Pulse Pressure
        subtraction_sub_type = "calc_1step_subtraction"
        question_template = "Calculate the pulse pressure at {charttime}?"

        calc_1step_subtraction_df = template_df.copy()
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records = patient_records.loc[len(patient_records)//3:].sample(n=1)
            question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
            answer = patient_records['sbp'].values[0] - patient_records['dbp'].values[0]
            calc_1step_subtraction_df.loc[len(calc_1step_subtraction_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":answer, 
                "answer_index":patient_records.index.values[0],
                "type":calculation_type, 
                "sub_type":subtraction_sub_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {subtraction_sub_type}: ", calc_1step_subtraction_df['question'].values[0])
        return calc_1step_subtraction_df

    def create_multiplication():
        # Multiplication - Temperature conversion factor
        # first step of Fahrenheit conversion
        multiplication_sub_type = "calc_1step_multiplication"
        question_template = "What is the value when multiplying the temperature at {charttime} by 1.8, round to 1 decimal places?"

        calc_1step_multiplication_df = template_df.copy()
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records = patient_records.loc[len(patient_records)//3:].sample(n=1)
            question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
            answer = round(patient_records['temperature'].values[0] * 1.8, 1)
            calc_1step_multiplication_df.loc[len(calc_1step_multiplication_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":answer, 
                "answer_index":patient_records.index.values[0],
                "type":calculation_type, 
                "sub_type":multiplication_sub_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {multiplication_sub_type}: ", calc_1step_multiplication_df['question'].values[0])
        return calc_1step_multiplication_df

    def create_division():
        #### Division - Shock Index
        division_sub_type = "calc_1step_division"
        question_template = "Calculate the Shock Index at {charttime}, round to 1 decimal places."

        calc_1step_division_df = template_df.copy()
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records = patient_records.loc[len(patient_records)//3:].sample(n=1)
            question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
            answer = patient_records['heartrate'].values[0] / patient_records['sbp'].values[0]
            calc_1step_division_df.loc[len(calc_1step_division_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":round(answer, 1), 
                "answer_index":patient_records.index.values[0],
                "type":calculation_type, 
                "sub_type":division_sub_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {division_sub_type}: ", calc_1step_division_df['question'].values[0])
        return calc_1step_division_df

    sub_calculation_type = "calculation_1step"
    calc_1step_df = pd.concat([create_addition()[:50], create_subtraction()[:50], create_multiplication()[:50], create_division()[:50]], ignore_index=True)
    print(f"Distribution of {sub_calculation_type}: ", calc_1step_df['sub_type'].value_counts())
    
    # convert to final format
    formatted_calc_1step_df = format_df(calc_1step_df)
    # Save to csv file
    formatted_calc_1step_df.to_csv(os.path.join(sub_out_dir, f"{sub_calculation_type}.csv"), index=False)
    print(f"Saved {sub_calculation_type} data with {len(formatted_calc_1step_df)} samples")
    

def create_2step_data(df, df_verbal, template_df, sub_out_dir):
    calc_2step_type = "calc_2step"

    def create_template1():
        #### Addition + Division - average of respiratory rate in first/last 2 records
        calc_2step_template1_type = f"{calc_2step_type}_addition_division"

        question_template = "Calculate the average of respiratory rate at the {case} 2 records, round to 1 decimal places."
        cases = ['first', 'last']

        calc_2step_template1_df = template_df.copy()
        vitals_counts = []
        for case in cases:
            patients = df[df['vitals_count']>3].groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(3, len(x)), random_state=RANDOM_STATE))['subject_id'].unique()
            for patient in patients:
                patient_records = df[df['subject_id']==patient]
                question = question_template.format(case=case)
                answer = patient_records['resprate'].values[:2].mean().round(1) if case == 'first' else patient_records['resprate'].values[-2:].mean().round(1)
                calc_2step_template1_df.loc[len(calc_2step_template1_df)] = {
                    "subject_id":patient, 
                    "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                    "question":question, 
                    "answer":round(answer, 1), 
                    "answer_index":f"{case} 2 records",
                    "type":calculation_type, 
                    "sub_type":calc_2step_template1_type}
                vitals_counts.append(patient_records['vitals_count'].values[0])

        calc_2step_template1_df = calc_2step_template1_df[:50]
        print(f"Example question of {calc_2step_template1_type}: ", calc_2step_template1_df['question'].values[0])
        return calc_2step_template1_df

    def create_template2():
        #### Subtraction + Division - a part of Mean Arterial Pressure (sbp-dbp)/3
        calc_2step_template2_type = f"{calc_2step_type}_subtraction_division"
        question_template = "Calculate a third of the pulse pressure at {charttime}, round to 1 decimal places."

        calc_2step_template2_df = template_df.copy()
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records = patient_records.loc[len(patient_records)//3:].sample(n=1)
            question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
            answer = round((patient_records['sbp'].values[0] - patient_records['dbp'].values[0])/3, 1)
            calc_2step_template2_df.loc[len(calc_2step_template2_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":answer, 
                "answer_index":patient_records.index.values[0],
                "type":calculation_type, 
                "sub_type":calc_2step_template2_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {calc_2step_template2_type}: ", calc_2step_template2_df['question'].values[0])
        return calc_2step_template2_df

    def create_template3():
        #### Addition + Multiplication - Temperature to Fahrenheit (°C × 1.8) + 32
        calc_2step_template3_type = f"{calc_2step_type}_addition_multiplication"
        question_template = "Convert temperature at {charttime} from Celsius to Fahrenheit, round to 1 decimal places."

        calc_2step_template3_df = template_df.copy()
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records = patient_records.loc[len(patient_records)//3:].sample(n=1)
            question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
            answer = round((patient_records['temperature'].values[0] * 1.8 + 32), 1)
            calc_2step_template3_df.loc[len(calc_2step_template3_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":answer, 
                "answer_index":patient_records.index.values[0],
                "type":calculation_type, 
                "sub_type":calc_2step_template3_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {calc_2step_template3_type}: ", calc_2step_template3_df['question'].values[0])
        return calc_2step_template3_df

    def create_template4():
        #### Subtraction + Multiplication - Percentage deviation
        calc_2step_template4_type = f"{calc_2step_type}_subtraction_multiplication"
        question_template = "Calculate the percentage that heart rate at {charttime} exceeds baseline (70 bpm), round to 1 decimal places."

        calc_2step_template4_df = template_df.copy()
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(5, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records = patient_records.loc[len(patient_records)//3:].sample(n=1)
            question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
            answer = round((patient_records['heartrate'].values[0] - 70) / 70 * 100, 1)
            calc_2step_template4_df.loc[len(calc_2step_template4_df)] = {
                "subject_id":patient, 
                "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                "question":question, 
                "answer":answer, 
                "answer_index":patient_records.index.values[0],
                "type":calculation_type, 
                "sub_type":calc_2step_template4_type}
            vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {calc_2step_template4_type}: ", calc_2step_template4_df['question'].values[0])
        return calc_2step_template4_df

    sub_calculation_type = "calculation_2step"
    calc_2step_df = pd.concat([create_template1()[:50], create_template2()[:50], create_template3()[:50], create_template4()[:50]], ignore_index=True)
    print(f"Distribution of {sub_calculation_type}: ", calc_2step_df['sub_type'].value_counts())
    
    # convert to final format
    formatted_calc_2step_df = format_df(calc_2step_df)
    # Save to csv file
    formatted_calc_2step_df.to_csv(os.path.join(sub_out_dir, f"{sub_calculation_type}.csv"), index=False)
    print(f"Saved {sub_calculation_type} data with {len(formatted_calc_2step_df)} samples")


def create_3step_data(df, df_verbal, template_df, sub_out_dir):
    calc_3step_type = "calc_3step"
    def create_template1():
        #### Subtraction + Division + Addition - Mean Arterial Pressure `(sbp-dbp)/3+dbp`
        calc_3step_template1_type = f"{calc_3step_type}_map"
        question_template = "Calculate the Mean Arterial Pressure at {charttime}, round to 1 decimal places."

        calc_3step_template1_df = template_df.copy()
        vitals_counts = []
        for patient in df.groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(6, len(x)), random_state=RANDOM_STATE))['subject_id'].unique():
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records_2 = patient_records.loc[len(patient_records)//3:].sample(n=min(2, len(patient_records)))
            for i in [0,1] if len(patient_records_2) > 1 else [0]:
                patient_records = patient_records_2.iloc[i:i+1]
                question = question_template.format(charttime=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '))
                answer = round((patient_records['sbp'].values[0] - patient_records['dbp'].values[0])/3 + patient_records['dbp'].values[0], 1)
                if len(calc_3step_template1_df) == 100:
                    break
                calc_3step_template1_df.loc[len(calc_3step_template1_df)] = {
                    "subject_id":patient, 
                    "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                    "question":question, 
                    "answer":answer, 
                    "answer_index":patient_records.index.values[0],
                    "type":calculation_type, 
                    "sub_type":calc_3step_template1_type}
                vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {calc_3step_template1_type}: ", calc_3step_template1_df['question'].values[0])
        return calc_3step_template1_df

    def create_template2():
        #### Subtraction + Division + Multiplication - Percentage deviation from baseline
        calc_3step_template2_type = f"{calc_3step_type}_deviation_percentage"
        question_template = "Calculate the percentage change in heart rate from {charttime_1}  to {charttime_2}, round to 2 decimal places."

        calc_3step_template2_df = template_df.copy()
        vitals_counts = []
        for patient in df[df['vitals_count']>3].groupby(["vitals_count"], group_keys=False).apply(lambda x: x.sample(n=min(12, len(x)), random_state=RANDOM_STATE))['subject_id'].unique()[:50]:
            patient_records = df[df['subject_id']==patient].reset_index(drop=True)
            patient_records_2 = patient_records.loc[min(len(patient_records)//3, len(patient_records)-2):].sample(n=min(3, len(patient_records))).sort_index()
            for i in [0,1] if len(patient_records_2) > 2 else [0]:
                patient_records = patient_records_2.iloc[i:i+2]
                question = question_template.format(charttime_1=str(patient_records['charttime'].values[0]).split('.')[0].replace('T', ' '), charttime_2=str(patient_records['charttime'].values[1]).split('.')[0].replace('T', ' '))
                answer = round((patient_records['heartrate'].values[1] - patient_records['heartrate'].values[0]) / patient_records['heartrate'].values[0] * 100, 2)
                calc_3step_template2_df.loc[len(calc_3step_template2_df)] = {
                    "subject_id":patient, 
                    "context":df_verbal[df_verbal['subject_id']==patient]['standard_verbal'].values[0], 
                    "question":question, 
                    "answer":answer, 
                    "answer_index":f"{patient_records.index.values[0]}, {patient_records.index.values[1]}",
                    "type":calculation_type, 
                    "sub_type":calc_3step_template2_type}
                vitals_counts.append(patient_records['vitals_count'].values[0])

        print(f"Example question of {calc_3step_template2_type}: ", calc_3step_template2_df['question'].values[0])
        return calc_3step_template2_df

    sub_calculation_type = "calculation_3step"
    calc_3step_df = pd.concat([create_template1(), create_template2()], ignore_index=True)
    print(f"Distribution of {sub_calculation_type}: ", calc_3step_df['sub_type'].value_counts())
    
    # convert to final format
    formatted_calc_3step_df = format_df(calc_3step_df)
    # Save to csv file
    formatted_calc_3step_df.to_csv(os.path.join(sub_out_dir, f"{sub_calculation_type}.csv"), index=False)
    print(f"Saved {sub_calculation_type} data with {len(formatted_calc_3step_df)} samples")


def create_arthmetic_data(df, df_verbal, template_df, output_dir):
    """
    Create 3 sub-types of arithmetic data:
    1. 1-step arithmetic
    2. 2-step arithmetic
    3. 3-step arithmetic
    
    Args:
        df: DataFrame with vitalsigns records, to get value for determistic answer programmatically
        df_verbal: DataFrame with chronological verbalized vitalsigns, to get context and question
        template_df: DataFrame to store the template data
        output_dir: Directory to store the final data

    Return:
        None with csv files saved in output_dir with format from format_df function
    """

    sub_out_dir = os.path.join(output_dir, calculation_type)
    os.makedirs(sub_out_dir, exist_ok=True)
    
    create_1step_data(df, df_verbal, template_df, sub_out_dir)
    create_2step_data(df, df_verbal, template_df, sub_out_dir)
    create_3step_data(df, df_verbal, template_df, sub_out_dir)
    