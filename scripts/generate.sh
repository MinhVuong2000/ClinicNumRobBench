for data_type in structured; do #  structured templated natural templated_variants padding_templated ablation_var_sum ablation_var_comparison ablation_var_calc ablation_var_retrieval
    input_dir="data/$data_type"
    output_dir="outputs/$data_type"
    for task in retrieval calculation comparison summary; do
        # # COT prompt
        # # unsloth/DeepSeek-R1-Distill-Llama-8B
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model unsloth/DeepSeek-R1-Distill-Llama-8B --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --level $task --dtype bf16 --max_tokens 16384 --trust-remote-code True
        # # huatuo-o1-8b
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model FreedomIntelligence/HuatuoGPT-o1-8B --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --batch-size 1 --level $task --prompt-type zero_shot_cot --max_tokens 16384
        # # qwen-2.5-7b-instruct
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model Qwen/Qwen2.5-7B-Instruct --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384

        # # meditron3-8b
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model OpenMeditron/Meditron3-8B --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --batch-size 1 --level $task --prompt-type zero_shot_cot --max_tokens 16384
        # # medphi-instruct
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model microsoft/MediPhi-Instruct --temperature 0.6 --input-dir $input_dir --output_dir $output_dir --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384
        # # ultra-medical-3.1-8b
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model TsinghuaC3I/Llama-3.1-8B-UltraMedical --input-dir $input_dir --output_dir $output_dir --temperature 0.7 --top_p 0.9 --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384
        # # meditron3-qwen2.5-7b
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model OpenMeditron/Meditron3-Qwen2.5-7B --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384

        # # qwen3-8b
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=40G python3.11 clinicnumrobbench/generate_llm_response.py --model Qwen/Qwen3-8B --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384
        # # qwen3-8b non-reasoning
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=40G python3.11 clinicnumrobbench/generate_llm_response.py --model Qwen/Qwen3-8B --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384 --reasoning-effort none
        # # medi-gemma-4b-it
        # CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model google/medgemma-4b-it --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --dtype bf16 --level $task --max_tokens 4096 --trust-remote-code True
        # # gemma3-4b-it
        # CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=24G python3.11 clinicnumrobbench/generate_llm_response.py --model google/gemma-3-4b-it --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384 --trust-remote-code True --dtype bf16
        # # llama-3.1-8b-instruct
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model meta-llama/Llama-3.1-8B-Instruct --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 1 --level $task --max_tokens 16384
        # # phi-3.5-mini
        # srun --partition=SCT --time=23:00:00 --gres=gpu:1 --mem=32G python3.11 clinicnumrobbench/generate_llm_response.py --model microsoft/Phi-3.5-mini-instruct --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --prompt-type zero_shot_cot --batch-size 2 --level $task --max_tokens 16384 --trust-remote-code True
        # # gpt-4.1
        # python3.11 clinicnumrobbench/generate_llm_response.py --model gpt-4.1 --temperature 0.6 --batch-size 4 --level $task --input-dir $input_dir --output_dir $output_dir --provider openai --prompt-type zero_shot_cot --max_tokens 16384
        # # gpt-5
        # python3.11 clinicnumrobbench/generate_llm_response.py --model gpt-5 --temperature 0.6 --batch-size 2 --level $task --input-dir $input_dir --output_dir $output_dir --provider openai --prompt-type zero_shot_cot --max_tokens 16384
        # # gpt-4.1-mini
        python3.11 clinicnumrobbench/generate_llm_response.py --model gpt-4.1-mini --temperature 0.6 --batch-size 4 --input-dir $input_dir --output_dir $output_dir --provider openai --prompt-type zero_shot_cot --max_tokens 16384 --level $task

        # BEDROCK
        # # gemma-3-27b
        # python3.11 clinicnumrobbench/generate_llm_response.py --provider bedrock --model google.gemma-3-27b-it --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --batch-size 4 --level $task --prompt-type zero_shot_cot --max_tokens 16384
        # # gemma-3-4b
        # python3.11 clinicnumrobbench/generate_llm_response.py --provider bedrock --model google.gemma-3-4b-it --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --batch-size 4 --level $task --prompt-type zero_shot_cot --max_tokens 16384
        # # qwen-3-8b
        # python3.11 clinicnumrobbench/generate_llm_response.py --provider bedrock --model qwen.qwen3-8b-instruct-v1:0 --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --batch-size 4 --level $task --prompt-type zero_shot_cot --max_tokens 16384
        # llama-3.3-70b-instruct
        # python3.11 clinicnumrobbench/generate_llm_response.py --provider bedrock --model meta.llama3-3-70b-instruct-v1:0 --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --batch-size 4 --level $task --prompt-type zero_shot_cot --max_tokens 16384
        # openai.gpt-oss-20b-1:0
        # python3.11 clinicnumrobbench/generate_llm_response.py --provider bedrock --model openai.gpt-oss-20b-1:0 --input-dir $input_dir --output_dir $output_dir --temperature 0.6 --batch-size 4 --level $task --prompt-type zero_shot_cot --max_tokens 16384
    done
done
