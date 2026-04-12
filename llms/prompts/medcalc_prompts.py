import json
from typing import List, Dict
import pandas as pd

from llms.prompts.base_prompt import BasePrompt

class MedCalcPrompts(BasePrompt):

    @classmethod
    def zero_shot(cls, note, question):
        system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}.'
        user_temp = f'Here is the patient note:\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
        return system_msg, user_temp

    @classmethod
    def one_shot(cls, note, question, example_note, example_output):
        system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
        system_msg += f'Here is an example patient note:\n\n{example_note}'
        system_msg += f'\n\nHere is an example task:\n\n{question}'
        system_msg += f'\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(value which is the answer to the question)}}:\n\n{json.dumps(example_output)}'
        user_temp = f'Here is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
        return system_msg, user_temp

    @classmethod
    def prepare_prompt(cls, df_row: pd.Series, prompt_type: str = "zero_shot") -> List[Dict[str, str]]:
        note = df_row["Patient Note"]
        question = df_row["Question"]
        if prompt_type == "zero_shot":
            system_msg, user_temp = cls.zero_shot(note, question)
        elif prompt_type == "one_shot":
            with open("data/one_shot_finalized_explanation.json", "r") as f:
                one_shot_json = json.load(f)
                calculator_id = str(df_row["Calculator ID"])
                example = one_shot_json[calculator_id]
            system_msg, user_temp = cls.one_shot(note, question, example["Patient Note"], {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]})
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_temp},
        ]
