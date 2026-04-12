def zero_shot_direct_answer_general_prompt(question: str, task_type: str = "numeracy"):
    system_msg = 'You are a helpful clinical assistant, answering the question directly and concisely. Your final output should only contain a JSON dict formatted as {"answer": <short_and_direct_answer_of_the_question>}.'
    user_temp = f'Here is the task:\n{question}'
    return system_msg, user_temp

def zero_shot_direct_answer_prompt(question: str, task_type: str = "numeracy"):
    system_msg = f'You are a helpful clinical assistant in {task_type}, answering the question directly and concisely. Your final output should only contain a JSON dict formatted as {{"answer": <short_and_direct_answer_of_the_question>}}.'
    user_temp = f'Here is the task:\n{question}'
    return system_msg, user_temp


def zero_shot_cot_prompt(question: str, task_type: str = "numeracy"):
    system_msg = f'You are a helpful clinical assistant in {task_type}. Please think step-by-step to solve the question. Your final output should end with a JSON dict formatted as {{"answer": <short_and_direct_answer_of_the_question>}}.'
    user_temp = f"Here is the task:\n{question}\n\nLet's think step-by-step to solve the question."
    return system_msg, user_temp


def zero_shot_cot_in_ans_prompt(question: str, task_type: str = "numeracy"):
    system_msg = f'You are a helpful clinical assistant in {task_type}. Please think step-by-step to solve the question. Your final output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
    user_temp = f"Here is the task:\n{question}\n\nLet's think step-by-step to solve the question."
    return system_msg, user_temp


def one_shot_direct_answer_prompt(question: str, example: str, task_type: str = "numeracy"):
    raise NotImplementedError("Not implemented yet")


def one_shot_cot_prompt(question: str, example: str, task_type: str = "numeracy"):
    raise NotImplementedError("Not implemented yet")


def prepare_prompt(question: str, prompt_type: str, task_type: str = "numeracy"):
    if prompt_type == "zero_shot_direct_answer":
        system_msg, user_temp = zero_shot_direct_answer_prompt(question, task_type)
    elif prompt_type == "zero_shot_cot":
        system_msg, user_temp = zero_shot_cot_prompt(question, task_type)
    elif prompt_type == "zero_shot_cot_in_ans":
        system_msg, user_temp = zero_shot_cot_in_ans_prompt(question, task_type)
    elif prompt_type == "zero_shot_direct_answer_general":
        system_msg, user_temp = zero_shot_direct_answer_general_prompt(question, task_type)
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_temp},
    ]

TEMPLATED_VARIANTS_PROMPT = """You are an expert of a practicing clinical doctor.

Your task is to interpret the provided vital sign record data and generate a rephrased clinical note based on provided data.

[Vital Sign Record Data Description] Patient's aperiodic vital signs is provided in JSON format. This JSON document contains (i) patient information such as age, gender; (ii) chronological vital signs during their stay such as recorded time, temperate (temperature), heart rate (heartrate),  respiratory rate (resprate), oxygen saturation (o2sat), systolic blood pressure (sbp), and diastolic blood pressure (dbp). All clinical events are ordered ascending by time.

[Rephrasing Guidelines] The generated clinical note must conform to these guidelines:
- Ground all information in the context of the provided data and ensure they are relevant to the patient's specific case.
- Remain all records in context.
- Context should follow professional language style as clinical note, paragraph.
- Clinical events are ordered ascending by time, but the order of information in each event should be shuffled with dynamic and robust expression while preserving all original information.
- Terms should be paraphrased in dynamic realistic usage of clinical practitioners.
- No generating generic or irrelevant text or comments.

Review the provided data and generate a rewording note that meet all outlined guidelines."""
