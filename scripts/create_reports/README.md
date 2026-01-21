# Report Generation Scripts

**Location**: `scripts/create_reports/`  
**Purpose**: Aggregate and summarize experimental results

---

## 📊 Available Scripts

### 1. `aggregate_complete_results.py`
**Primary aggregation script** - Use this for final results compilation.

**Purpose**:
- Collects ALL evaluated experiments (original + corrected models)
- Generates comprehensive comparison reports
- Exports both CSV and Markdown formats

**When to run**:
```
✅ After ALL evaluations complete
✅ Final step in the workflow
❌ Do NOT run during training
```

**Workflow position**:
```
Step 1: Train models          → run_all_cada_experiments.sh
                                (or use specific run_train_*.sh scripts)
Step 2: Evaluate models       → evaluate_all_cada_experiments.sh
► Step 3: RUN THIS SCRIPT     → aggregate_complete_results.py ◄
```

**Usage**:
```bash
cd /home/ngoductam/KLGrade
python scripts/create_reports/aggregate_complete_results.py
```

**Output**:
- `runs/kiocmil_cada/complete_results.csv` - Raw data
- `runs/kiocmil_cada/COMPLETE_RESULTS.md` - Formatted report
- `docs/experiments/CADA_COMPLETE_RESULTS.md` - Documentation copy

---

## 🗑️ Deprecated Scripts (DO NOT USE)

The following scripts in the project root are **deprecated** and have been superseded:

### ❌ `collect_cada_results.py` (root directory)
- **Status**: DEPRECATED
- **Reason**: Only collects training results, not evaluations
- **Replaced by**: `aggregate_complete_results.py`
- **Action**: Can be deleted

### ❌ `aggregate_cada_evaluations.py` (root directory)  
- **Status**: DEPRECATED
- **Reason**: Only handles original models, not corrected versions
- **Replaced by**: `aggregate_complete_results.py`
- **Action**: Can be deleted

---

## 📝 Script Comparison

| Feature | collect_cada_results.py | aggregate_cada_evaluations.py | aggregate_complete_results.py |
|---------|-------------------------|-------------------------------|-------------------------------|
| **Status** | ❌ Deprecated | ❌ Deprecated | ✅ ACTIVE |
| **Scope** | Training metrics only | Evaluation (original) | ALL (original + corrected) |
| **Output** | CSV only | MD + CSV | MD + CSV + docs copy |
| **Derived metrics** | ❌ No | ✅ Yes | ✅ Yes |
| **Comparison** | ❌ No | ❌ No | ✅ Original vs Corrected |
| **When to use** | Never | Never | After final evaluation |

---

## 🔄 Standard Workflow

### Complete Training & Evaluation Pipeline

```bash
# 1. Train all models (20 experiments)
bash scripts/training/run_all_cada_experiments.sh

# Or train selectively:
bash scripts/training/run_train_4_5_class.sh
bash scripts/training/run_train_8_10_class.sh
bash scripts/training/run_processed_4_8_class.sh
bash scripts/training/run_processed_5_10_class.sh

# 2. Evaluate all trained models
bash scripts/training/evaluate_all_cada_experiments.sh

# 3. Aggregate results
python scripts/create_reports/aggregate_complete_results.py
```

---

## 📂 Output Structure

After running `aggregate_complete_results.py`:

```
runs/kiocmil_cada/
├── complete_results.csv           # All 30 experiments data
├── COMPLETE_RESULTS.md            # Formatted comparison report
└── [experiment_dirs]/
    └── evaluation/
        └── metrics.json           # Source data

docs/experiments/
└── CADA_COMPLETE_RESULTS.md       # Documentation copy
```

---

## 🧹 Cleanup Recommendations

To clean up deprecated scripts:

```bash
cd /home/ngoductam/KLGrade

# Move deprecated scripts to archive (optional)
mkdir -p archive/deprecated_scripts
mv collect_cada_results.py archive/deprecated_scripts/
mv aggregate_cada_evaluations.py archive/deprecated_scripts/

# Or delete them directly
# rm collect_cada_results.py aggregate_cada_evaluations.py
```

---

## ✅ Best Practices

1. **Always run evaluation first** before aggregation
2. **Use only `aggregate_complete_results.py`** for result compilation
3. **Check output files** exist before running aggregation:
   ```bash
   find runs/kiocmil_cada -name "metrics.json" | wc -l
   # Should show 30 (20 original + 10 corrected)
   ```
4. **Version results** by copying to timestamped backups if needed

---

**Last Updated**: 2026-01-21  
**Maintainer**: Project Team  
**Status**: Production Ready
