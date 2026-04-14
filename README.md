# CAML: Consensus Aggregation of Molecular Models

## Project Overview

CAML (Consensus Aggregation of Molecular Models) is a framework for fusing multiple molecular property generation models, designed to generate compounds that simultaneously meet multiple molecular property targets by optimizing the weights and parameters of different models.

This project primarily addresses the challenge of single models struggling to simultaneously optimize multiple molecular properties. By intelligently fusing multiple models specialized for different properties, it enables more comprehensive molecular design.

## Project Structure

```
CAML-main/
├── CAML.py              # Main fusion implementation
├── CAML.yml             # Environment configuration file
├── src/                 # Source code directory
│   ├── CAML_utils.py    # Core utility functions
│   ├── task_vectors.py  # Task vector implementation
│   ├── model/           # Model-related code
│   ├── data/            # Data processing
│   ├── utils/           # General utilities
│   └── prop/            # Property evaluation
├── model/               # Model directory
│   ├── chemgpt_pretrained_selfies/  # Pretrained model
│   ├── lit_chemgpt.py   # Model implementation
│   └── run_train.py     # Training script
└── prop/                # Property evaluation related files
    ├── sascorer.py      # Synthetic accessibility scoring
    └── jnk3_gsk_scorer.py  # Biological activity scoring
```

## Installation and Environment Configuration

### Dependencies

- Python 3.9+
- Torch 2.8.0
- Transformers
- RDKit
- NumPy
- Pandas
- CMA-ES
- NetworkX
- selfies 2.1.1

### Environment Setup

```bash
# Create environment using conda
conda env create -f CAML.yml

# Activate environment
conda activate CAML
```

## Usage

### Basic Usage

1. **Prepare pretrained and fine-tuned models**

   Ensure that the pretrained model and models fine-tuned for different properties are prepared, and set the correct paths in CAML.py:

   ```python
   pretrain_model_path = "/path/to/pretrained/model"
   ft_model_path = [f"/path/to/finetuned/model/{prop_name}_RL" for prop_name in prop_expert]
   ```

2. **Run fusion**

   Directly run the CAML.py file:

   ```bash
   python CAML.py
   ```

   This will start the CMA-ES optimization process, fuse multiple models, and generate optimized molecules.

### Configuration Parameters

In the `run_CAML` function, you can adjust the following parameters:

- `generations`: Number of optimization iterations
- `popsize`: Population size
- `sigma_init`: Initial search step size
- `n_mols`: Number of molecules generated per evaluation
- `merge_func`: Fusion strategy ("sum", "mean", "max")
- `props`: List of properties to optimize
- `baseline_dict`: Baseline values for properties

### Custom Properties

To add new molecular property evaluations, you can add the corresponding evaluation logic in the `reward_fn` function in `src/CAML_utils.py`.

## Core Algorithms

### 1. Task Vector Calculation

```python
tv_flat_checks = flat_ft - flat_ptm
```

Calculate the parameter differences between fine-tuned models and the pretrained model to form task vectors.

### 2. Model Merging

```python
def merge_with_nash_merging(task_deltas, alphas, ks, merge_func="sum", device=None):
    # Each task performs top-k individually
    updated_list = []
    for i in range(K):
        masked, _ = topk_values_mask(all_checks[i].unsqueeze(0), K=float(ks[i]))
        updated_list.append(masked.squeeze(0))
    updated = torch.stack(updated_list, dim=0)
    
    # Aggregation
    merged = disjoint_merge_wosign(updated, merge_func, alphas)
    fused = ptm_vec + merged
    
    return fused
```

### 3. Parameter Optimization

Use the CMA-ES algorithm to optimize fusion parameters (ks and alphas):

```python
es = CMAEvolutionStrategy(mean_init, sigma_init, {
    "popsize": popsize,
    "bounds": bounds,
    'seed': 42,
})
```

### 4. Multi-Property Balancing

Use Nash bargaining theory to balance multiple property objectives:

```python
w = np.exp(-beta_weight * agg)
w = w / (np.sum(w) + eps)
weighted_log_prod = float(np.sum(w * np.log(agg + eps)))
```

## Output Results

After running, the system will generate the following outputs:

1. **Best fused model**: Saved as `best_nash_model.pt`
2. **Generated molecules**: Saved as `best_nash_smiles.csv`
3. **Optimization history**: Saved as `nash_merging_Regularization_history.json`

## Example Output

```
[Gen 000] best_fit=-0.6789
  k=[0.7 0.7]
  alpha=[0.5 0.5]
  Real Scores=[0.7543 0.6218]
  σ=0.3000

[Gen 001] best_fit=-0.6543
  k=[0.72 0.68]
  alpha=[0.55 0.45]
  Real Scores=[0.7612 0.6345]
  σ=0.2800
...
Saving best model to best_nash_model.pt...
Model saved.
Saving best SMILES to best_nash_smiles.csv...
SMILES saved.
```

## Citation

If you use CAML in your research, please cite the following work:

```
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.