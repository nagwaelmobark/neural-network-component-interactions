# Neural Network Component Interactions - CIF Framework

## Research Paper

**"A Framework for Understanding Neural Network Component Interactions: Selection Principles, Guidelines, and Empirical Evidence"**

**Author:** Nagwa Elmobark
**Affiliation:** Department of Computer Science, University of Mansoura, Egypt
**GitHub:** [github.com/nagwaelmobark](https://github.com/nagwaelmobark)

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `nn_component_interactions.py` | Main experiment code (Table 4 & Table 5) |
| `Figure4_Experiment1_Convergence.png` | Activation-Initialization convergence analysis |
| `Figure5_Experiment2_Convergence.png` | Normalization-Optimizer convergence analysis |
| `Table1_Results.csv` | Raw results - Experiment 1 (Table 4 in paper) |
| `Table2_Results.csv` | Raw results - Experiment 2 (Table 5 in paper) |
| `requirements.txt` | Python dependencies |

---

## 🔬 Experiments

### Experiment 1: Activation × Initialization (Table 4)

| Configuration | Epochs to 90% | Final Accuracy |
|--------------|---------------|----------------|
| **GELU + He** | **21** | **91.8%** |
| **ReLU + He** | **23** | **91.2%** |
| Leaky ReLU + He | 22 | 91.5% |
| GELU + Xavier | 29 | 90.5% |
| Leaky ReLU + Xavier | 58 | 89.1% |
| ReLU + Xavier | 67 | 88.3% |
| ReLU + Random | 89 | 85.7% |
| Tanh + Xavier | 41 | 87.9% |
| Tanh + He | 124 | 76.2% |
| Sigmoid + Xavier | 156 | 68.4% |

**Key Finding:** Proper activation-initialization pairing achieves up to 2.9× faster convergence for typical pairs.

---

### Experiment 2: Normalization × Optimizer (Table 5)

| Configuration | Epochs to 90% | Final Accuracy | Training Time |
|--------------|---------------|----------------|---------------|
| **BatchNorm + Adam** | **19** | **90.8%** | 1.15× |
| **BatchNorm + SGD** | **23** | **91.2%** | 1.0× |
| LayerNorm + Adam | 22 | 90.3% | 1.18× |
| LayerNorm + SGD | 37 | 88.5% | 1.05× |
| None + Adam | 48 | 87.9% | 1.08× |
| None + SGD | 142 | 83.1% | 0.95× |

**Key Finding:** Normalization enables up to 6.2× faster training compared to no normalization.

---

## ⚙️ Experimental Setup

- **Dataset:** CIFAR-10 (50K train / 10K test)
- **Architecture:** ResNet-18 (11.2M parameters)
- **Batch Size:** 128
- **Epochs:** 100
- **SGD:** lr=0.1, momentum=0.9, weight_decay=5e-4
- **Adam:** lr=0.001, weight_decay=5e-4
- **Data Augmentation:** RandomCrop (padding=4), RandomHorizontalFlip
- **Seeds:** 5 random seeds [42, 0, 1, 7, 123] — results reported as mean ± std
- **Hardware:** NVIDIA Tesla V100 (16GB)

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run experiments
```bash
python nn_component_interactions.py
```

This will run:
- **Table 4:** 10 activation-initialization configurations × 5 seeds (~3-5 hours on GPU)
- **Table 5:** 6 normalization-optimizer configurations × 5 seeds (~2-3 hours on GPU)

### 3. Outputs
- `table4_activation_init_results.csv`
- `table5_norm_optimizer_results.csv`

> **Tip:** To run a quick test, edit `SEEDS = [42]` at the top of the script.

---

## 📋 Reproducibility

- ✅ Random seeds fixed across all libraries (PyTorch, NumPy, Python random)
- ✅ All hyperparameters documented
- ✅ 5 independent runs per configuration
- ✅ Statistical significance: p < 0.001 (paired t-tests)
- ✅ Standard deviation < 2 epochs across seeds

---

## 📖 Citation

```bibtex
@article{elmobark2025cif,
  title={A Framework for Understanding Neural Network Component Interactions:
         Selection Principles, Guidelines, and Empirical Evidence},
  author={Elmobark, Nagwa},
  journal={The Journal of Supercomputing},
  year={2025}
}
```

---

## 📄 License

- **Code:** MIT License
- **Data:** CIFAR-10 — [original license](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 🙏 Acknowledgments

- CIFAR-10 dataset: Krizhevsky, A. (2009)
- ResNet: He et al. (2015)
- PyTorch framework
