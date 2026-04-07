# ClinicNumRobBench
Official Implementation of ["How Robust Are Large Language Models for Clinical Numeracy? An Empirical Study on Numerical Reasoning Abilities in Clinical Contexts"](#).

## Project Structure


## Environment Setup

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MinhVuong2000/ClinicNumRobBench.git
   cd ClinicNumRobBench
   ```

2. **Install Poetry (if not already installed):**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Create virtual environment:**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   poetry install
   ```

#### Install git-lfs as needed
```bash
wget https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-linux-amd64-v3.2.0.tar.gz
tar xvf git-lfs-linux-amd64-v3.2.0.tar.gz
cd git-lfs-3.2.0/
chmod +x install.sh
sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh
mkdir -p ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
./install.sh
git-lfs --version
cd ..
git lfs install
git lfs pull
```

## Build benchmark
### Format data and extract numeracy samples from MIMIC-IV ED & MIMIC-IV dataset
```bash
python data_creation/filter_num_data.py 2>&1 | tee data/log/filter_num_data.log
```
### Build data
```bash
python data_creation/create_structure.py 2>&1 | tee data/log/create_structure.log
python data_creation/create_semi_structure.py 2>&1 | tee data/log/create_semi_structure.log
python data_creation/create_padding_semi_structure.py 2>&1 | tee data/log/create_padding_semi_structure.log
python data_creation/create_unstructure.py 2>&1 | tee data/log/create_unstructure.log
```
Return data has format for each row:
- question
- answer
- data_source: original datasets & sub-category derived

## Run LLM to get responses
```bash
sh scripts/generate.sh
```

## Evaluate
### Baseline
```bash
sh scripts/eval.sh
```

## Analysis

## Citation
If you find this paper or the repo useful for your work, please consider citing the paper
