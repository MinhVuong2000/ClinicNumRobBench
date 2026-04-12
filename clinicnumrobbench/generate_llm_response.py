from math import nan
import os
import argparse
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

from const import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, TOGETHERAI_API_KEY, BEDROCK_API_KEY, BEDROCK_ENDPOINT
from llms.openai_llm import OpenAILLM
from llms.hf_causal_llm import HFCasualLLM
from llms.base_language_model import BaseLanguageModel
from llms.togetherai_llm import TogetherAILLM
from llms.bedrock_llm import BedrockLLM
from clinicnumrobbench.prompt import prepare_prompt


def _chunk_ranges(total: int, batch_size: int) -> List[range]:
    return [range(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]


def _build_messages(df: pd.DataFrame, idxs: range, task_type: str, prompt_type: str = "zero_shot_cot") -> List[List[Dict[str, str]]]:
    messages_list: List[List[Dict[str, str]]] = []
    for i in idxs:
        row = df.iloc[i]
        messages = prepare_prompt(row["question"], prompt_type=prompt_type, task_type=task_type)
        messages_list.append(messages)
    return messages_list


def load_model(model: str, dtype: str = "fp16", trust_remote_code: bool = True, provider: str = "hf"):
    """Load the model."""
    if "gpt-4.1" in model or "gpt-5" in model:
        return OpenAILLM(model_name=model, api_key=AZURE_OPENAI_API_KEY, api_base=AZURE_OPENAI_ENDPOINT)
    elif provider == "togetherai":
        return TogetherAILLM(model_name=model, api_key=TOGETHERAI_API_KEY)
    elif provider == "bedrock":
        return BedrockLLM(model_name=model, api_key=BEDROCK_API_KEY, api_base=BEDROCK_ENDPOINT)
    else:
        return HFCasualLLM(model_name=model, trust_remote_code=trust_remote_code, dtype=dtype)


def evaluate_batch(model: BaseLanguageModel, messages: List[List[Dict[str, str]]], max_tokens: int, temperature: float, top_p: float, top_k: int, verbose: bool, reasoning_effort: str, labels: List[Dict[str, Any]], return_logprobs: bool):
    """Evaluate the model on a batch of messages."""
    if isinstance(model, OpenAILLM) or isinstance(model, TogetherAILLM) or isinstance(model, BedrockLLM):
        return model.batch_generate_response(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            reasoning_effort=reasoning_effort,
            labels=labels,
            return_logprobs=return_logprobs,
        )
    elif isinstance(model, HFCasualLLM):
        return model.batch_generate_response(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            reasoning_effort=reasoning_effort,
            verbose=verbose,
            return_logprobs=return_logprobs,
        )
    else:
        raise ValueError(f"Model {model} not supported")


def main(input_dir: str, output_dir: str, level: str, model: str, prompt_type: str, temperature: float, top_p: float, top_k: int, max_tokens: int, batch_size: int, verbose: bool, reasoning_effort: str, num_samples: int, dtype: str, additional_cases: bool, return_logprobs: bool, provider: str, trust_remote_code: bool):
    """Main function to evaluate the model on the dataset."""
    data_dir = os.path.join(input_dir, level)

    model_config_str = f"{model.replace('/', '__')}_{temperature}_{top_p}_{top_k}_{reasoning_effort}"

    llm = load_model(model, dtype=dtype, provider=provider, trust_remote_code=trust_remote_code)

    files = [file for file in os.listdir(data_dir) if file.endswith(".csv")]
    num_tasks = len(files)
    if not additional_cases:
        files = [file for file in files if file not in [
            "temporal_sequence_retrieval.csv", "temporal_sequence_retrieval"
            "medcalc.csv", "medcalc"
        ]]
    for i, file in enumerate(files):
        print(f"Processing {file}: {i+1}/{num_tasks}")
        task_output_dir = os.path.join(output_dir, level, prompt_type, file.replace(".csv", ""))
        os.makedirs(task_output_dir, exist_ok=True)
        output_path = os.path.join(task_output_dir, f"{model_config_str}.csv")
        if os.path.exists(output_path):
            input_df = pd.read_csv(output_path)
        else:
            if num_samples > 0:
                input_df = pd.read_csv(os.path.join(data_dir, file)).iloc[:num_samples]
            else:
                input_df = pd.read_csv(os.path.join(data_dir, file))

        total = len(input_df)
        out_df = input_df.copy()
        range_idxs = _chunk_ranges(total, batch_size)

        if "generated_text" not in out_df.columns:
            out_df['generated_text'] = pd.Series([nan] * total, dtype="object")

        for idxs in tqdm(range_idxs, desc=f"Generating answers for {file} with {model_config_str}"):
            if out_df.loc[idxs, "generated_text"].notna().all():
                print(f"Skipping batch {idxs} because it already has predictions")
                continue
            messages = _build_messages(out_df, idxs, task_type=level, prompt_type=prompt_type)
            if 'num_input_tokens' in out_df.columns:
                labels = out_df.loc[idxs,["generated_text", "num_input_tokens","num_output_tokens","generation_time"]].to_dict(orient="records")
            else:
                labels = None
            result = evaluate_batch(llm, messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k, verbose=verbose, reasoning_effort=reasoning_effort, labels=labels, return_logprobs=return_logprobs)
            out_df.loc[idxs, "generated_text"] = result.get("generated_text")
            out_df.loc[idxs, "generation_time"] = result.get("generation_time")
            out_df.loc[idxs, "num_input_tokens"] = result.get("num_input_tokens")
            out_df.loc[idxs, "num_output_tokens"] = result.get("num_output_tokens")
            if return_logprobs:
                out_df.loc[idxs, "logprobs"] = [str(v) for v in result.get("logprobs")] if result.get("logprobs") is not None else None # convert to string to avoid numpy array error

            out_df.to_csv(output_path, index=False)
            print(f"Saved answers to {output_path} with {idxs[-1]+1}/{total} samples")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--level",
        type=str,
        default="lv1",
        choices=["lv1", "lv2", "lv3", "lv4", "lv4_full", "retrieval", "calculation", "comparison", "summary", "ablation_var_comparison"], # lv4_full is the full data of lv4
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        # choices=["gpt-4.1", "gpt-5", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "microsoft/Phi-4-reasoning", "FreedomIntelligence/HuatuoGPT-o1-8B", "TsinghuaC3I/Llama-3.1-8B-UltraMedical", "Qwen/QwQ-32B", "microsoft/MediPhi-Instruct","aaditya/Llama3-OpenBioLLM-8B", "Qwen/Qwen3-8B"],
    )
    argparser.add_argument(
        "--prompt-type",
        type=str,
        default="zero_shot_direct_answer",
        choices=["zero_shot_direct_answer", "zero_shot_cot", "zero_shot_direct_answer_general"],
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )
    argparser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
    )
    argparser.add_argument(
        "--top_k",
        type=int,
        default=40,
    )
    argparser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
    )
    argparser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    argparser.add_argument(
        "--verbose",
        type=bool,
        default=True,
    )
    argparser.add_argument(
        "--input-dir",
        type=str,
        default="data",
    )
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
    )
    argparser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
    )
    argparser.add_argument(
        "--num-samples",
        type=int,
        default=0,
    )
    argparser.add_argument(
        "--reasoning-effort",
        type=str,
        default="minimal",
        choices=["low", "medium", "high", "minimal", "none"],
    )
    argparser.add_argument(
        "--additional-cases",
        type=bool,
        default=False,
    )
    argparser.add_argument(
        "--return-logprobs",
        type=bool,
        default=False,
    )
    argparser.add_argument(
        "--provider",
        type=str,
        default="hf",
        choices=["togetherai", "openai", "hf", "bedrock"],
    )
    argparser.add_argument(
        "--trust-remote-code",
        type=bool,
        default=False,
    )
    args = argparser.parse_args()
    print(args)
    main(**args.__dict__)

#srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=16G python3.11 clinicnumrobbench/generate_llm_response.py --model Qwen/Qwen3-8B --temperature 0.6 --batch-size 4 --level lv3 --prompt-type zero_shot_cot
#srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=16G python3.11 clinicnumrobbench/generate_llm_response.py --model google/medgemma-4b-it --temperature 0.6 --batch-size 4 --level lv3 --prompt-type zero_shot_cot --dtype bf16 --provider togetherai
