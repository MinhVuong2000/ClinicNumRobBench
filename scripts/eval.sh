python3.11 clinicnumrobbench/eval.py --data_dir outputs/structured --prompt_type zero_shot_cot

# python3.11 clinicnumrobbench/eval.py --data_dir outputs/templated # default: --prompt_type zero_shot_direct_answer
# python3.11 clinicnumrobbench/eval.py --data_dir outputs/templated --prompt_type zero_shot_cot

# python3.11 clinicnumrobbench/eval.py --data_dir outputs/padding_templated --prompt_type zero_shot_cot

# python3.11 clinicnumrobbench/eval.py --data_dir outputs/templated_variants --prompt_type zero_shot_cot

# python3.11 clinicnumrobbench/eval.py --data_dir outputs/natural --prompt_type zero_shot_cot

# python3.11 clinicnumrobbench/eval.py --data_dir outputs/ablation_var_sum --prompt_type zero_shot_cot
# python3.11 clinicnumrobbench/eval.py --data_dir outputs/ablation_var_comparison --prompt_type zero_shot_cot

# python3.11 clinicnumrobbench/analysis/fine_grain_comparison_summary.py --prompt_type zero_shot_cot --data_dir outputs --task comparison
# python3.11 clinicnumrobbench/analysis/fine_grain_comparison_summary.py --prompt_type zero_shot_cot --data_dir outputs --task summary
