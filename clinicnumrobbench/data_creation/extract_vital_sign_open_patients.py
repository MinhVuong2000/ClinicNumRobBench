import re
import os
import json
import argparse
from typing import List
from json.decoder import JSONDecodeError
from tqdm import tqdm

import pandas as pd
from pydantic import BaseModel, Field
import datasets

from llms.hf_causal_llm import HFCasualLLM
from llms.openai_llm import OpenAILLM

from const import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT


class ValueObj(BaseModel):
    text: str|None = Field(description="Raw Text of the value if having, otherwise None, for example: ' BP 120/80 mmHg on test'")
    number: str|None = Field(description="Number of the value if having, otherwise None, for example: 120/80")
    unit: str|None = Field(description="Unit of the value if having, otherwise None, for example: mmHg")

class VitalSignExtractionObj(BaseModel):
    temperature: ValueObj
    heart_rate: ValueObj
    respiratory_rate: ValueObj
    oxygen_saturation: ValueObj
    blood_pressure: ValueObj

prompt_template = """
You are a medical documentation specialist. Your task is to extract raw vital sign text in given patient note which contains temperature, heart rate, respiratory rate, oxygen saturation, blood pressure. Text extracted must be a exact match from the patient note.
Output format follow the JSON schema:
{vital_sign_extraction_obj_schema}

Patient Note:
{patient_note}
"""

def get_full_text(description: str, extracted_vital_sign_json: dict) -> str:
    min_index = None
    max_index = None
    if not extracted_vital_sign_json:
        return ""
    for value in extracted_vital_sign_json.values():
        if isinstance(value, dict) and value['text'] is not None:
            value_index = description.find(value['text'])
            if value_index == -1: # not found, not process
                return ""
            value_end_index = value_index + len(value['text'])
            if min_index is None or value_index < min_index:
                min_index = value_index
            if max_index is None or value_end_index > max_index:
                max_index = value_end_index
    if min_index is None or max_index is None:
        return ""
    return description[min_index:max_index]

def extract_vital_sign_from_note(notes: List[str], model: HFCasualLLM, max_tokens: int = 16384) -> List[VitalSignExtractionObj|None]:
    messages = [[{"role": "user", "content": prompt_template.format(patient_note=note, vital_sign_extraction_obj_schema=VitalSignExtractionObj.model_json_schema())}] for note in notes]
    responses = model.batch_generate_response(messages, max_tokens=max_tokens, temperature=0.3, top_p=0.95, top_k=40, return_logprobs=False, reasoning_effort="high")
    extracted_texts = [response.split('</think>')[-1].replace("```json", "").replace("```", "").strip() for response in responses["generated_text"]]

    results = []
    for extracted_text in extracted_texts:
        try:
            results.append(VitalSignExtractionObj(**json.loads(extracted_text)))
        except JSONDecodeError:
            print(f"Error parsing extracted text: {extracted_text}")
            results.append(None)

    return results

def main(batch_size: int = 2, output_path: str = "data/open_patients_extracted_vital_sign.csv", model_name: str = "Qwen/Qwen3-8B", max_tokens: int = 16384):
    if 'gpt' in model_name:
        model = OpenAILLM(model_name=model_name, api_key=AZURE_OPENAI_API_KEY, api_base=AZURE_OPENAI_ENDPOINT)
    else:
        model = HFCasualLLM(model_name=model_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
    else:
        df = datasets.load_dataset("ncbi/Open-Patients", split="train").to_pandas()
        # filter out rows with rule
        vital_sign_pattern = 'vital sign|vitals|vitals sign'
        number_contain_pattern = r'(\d+[^\d]+\d*){5,}'
        year_old_pattern = 'year old|year-old|years old|years-old|yr-old|yr old'
        df = df[df['description'].str.lower().map(
            lambda note: (bool(re.search(vital_sign_pattern, note))
                and bool(re.search(number_contain_pattern, note))
                and bool(re.search(year_old_pattern, note))
                    ) )
        ] # 15K
    patient_notes = df['description']

    # Process notes in batches
    len_patient_notes = len(patient_notes)
    extracted_json_column_name = 'extracted_vital_sign_json'
    extracted_text_column_name = 'extracted_vital_sign_text'
    for i in tqdm(range(0, len_patient_notes, batch_size)):
        start_idx = i
        end_idx = min(i + batch_size, len_patient_notes)
        if extracted_text_column_name in df.columns and df.loc[start_idx:end_idx, extracted_text_column_name].notna().all():
            # tqdm.write(f"Skipping batch {start_idx}-{end_idx} because it already has extracted vital sign")
            continue

        if extracted_text_column_name not in df.columns:
            df[extracted_text_column_name] = None
        if extracted_json_column_name not in df.columns:
            df[extracted_json_column_name] = None

        batch_notes = patient_notes[start_idx:end_idx]
        batch_results = extract_vital_sign_from_note(batch_notes, model=model, max_tokens=max_tokens)
        df.loc[df.iloc[start_idx:end_idx].index, extracted_json_column_name] = [extracted_vital_sign.model_dump() if extracted_vital_sign else None for extracted_vital_sign in batch_results]
        df.loc[df.iloc[start_idx:end_idx].index, extracted_text_column_name] = df.iloc[start_idx:end_idx].apply(lambda r: get_full_text(r['description'], r[extracted_json_column_name]), axis=1)
        df.to_csv(output_path, index=False)
        # tqdm.write(f"Saved batch {end_idx}/{len_patient_notes} notes")

    tqdm.write(f"Completed processing {len_patient_notes} notes")

    # Convert string to dict if needed
    if isinstance(df[extracted_json_column_name][0], str):
        df[extracted_json_column_name] = df[extracted_json_column_name].map(lambda x: eval(x) if isinstance(x, str) else x)

    # Remove rows where all attributes in extracted_vital_sign_json are None
    df = df[df["extracted_vital_sign_json"].map(lambda x: not(isinstance(x, dict) and all(value['number'] is None for value in x.values())))]
    df.to_csv(output_path, index=False)
    print(f"Filtered out {len_patient_notes-len(df)} rows with no existing vital signs. Remaining: {len(df)} notes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="data/open_patients_extracted_vital_sign.csv")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(batch_size=args.batch_size, output_path=args.output_path, model_name=args.model_name, max_tokens=args.max_tokens)

# srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=40G python3.11 clinicnumrobbench/data_creation/extract_vital_sign_open_patients.py
# srun --partition=SCT --mem=1G python3.11 clinicnumrobbench/data_creation/extract_vital_sign_open_patients.py --model-name gpt-4.1-mini --batch-size 4
