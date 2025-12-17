# Conda – Practical Guide & Mental Model

This document summarizes what Conda is, how it works, and how to use it effectively, based on hands‑on learning during an ML bootcamp.

---

## What is Conda?

Conda is **both**:

1. A **package manager** (like `pip`, `npm`, or `brew`)
2. An **environment manager** (like `venv`, but more powerful)

Unlike `pip` + `venv`, Conda can manage:
- The **Python interpreter itself**
- **Native (non‑Python) libraries** such as OpenSSL, BLAS, CUDA
- Packages written in **multiple languages** (C, C++, Fortran)

Conda is widely used in **data science, ML, and scientific computing** where native dependencies matter.

---

## What Problem Conda Solves

Conda avoids dependency conflicts like:
- Different projects needing different Python versions
- Native libraries mismatching system versions
- Fragile `pip install` workflows that require compiling C/C++

Each Conda environment is a **self‑contained runtime**:

- Python version
- Python packages
- Native libraries
- Toolchain dependencies

Multiple environments can coexist on the same machine without conflicts.

---

## Conda vs Python `venv`

### Python `venv`

- Isolates **Python packages only**
- Uses the **system Python version**
- Does **not** manage native libraries
- Lightweight and fast

### Conda

- Isolates Python **and** native dependencies
- Manages the **Python interpreter version**
- Handles CUDA, OpenSSL, BLAS, etc.
- Uses a dependency solver for binary compatibility
- Heavier, but far more reproducible

### Summary Table

| Capability | venv | Conda |
|----------|------|-------|
| Python isolation | ✅ | ✅ |
| Python version management | ❌ | ✅ |
| Native libraries | ❌ | ✅ |
| Non‑Python packages | ❌ | ✅ |
| Reproducibility | Medium | High |

Rule of thumb:
- **Pure Python apps → venv**
- **ML / scientific stacks → Conda**

---

## What Does “conda activate” Mean?

Activating a Conda environment:
- Modifies the **shell session** (not starting a process)
- Changes `PATH` so `python`, `pip`, etc. come from the environment
- Sets environment variables like `CONDA_PREFIX`
- Configures native library paths

Activation is required so your shell uses the correct runtime.

You’ll see:

```bash
(fraud-detection) $
```

This is shell state, not a running service.

---

## Why `conda init` Is Required

`conda activate` relies on **shell functions**, not just a binary.

`conda init`:
- Adds Conda shell hooks to `~/.zshrc` or `~/.bashrc`
- Enables `conda activate` / `conda deactivate`

Fix activation errors with:

```bash
conda init zsh
exec zsh
```

---

## Checking Whether Conda Is Installed

Useful commands:

```bash
conda --version
which conda
conda info
```

Look for install directories:

```bash
~/anaconda3
~/miniconda3
```

---

## Installing Dependencies into an Active Environment

### Activate first

```bash
conda activate fraud-detection
```

Verify:

```bash
which python
```

---

### Preferred: Use Conda

```bash
conda install numpy pandas scikit-learn
```

Strongly recommended:

```bash
conda install -c conda-forge numpy pandas scikit-learn
```

---

### Using `pip` (when necessary)

Only use `pip` **after** Conda installs:

```bash
pip install shap imbalanced-learn
```

Rule of thumb:
> **Conda first, pip last**

---

## environment.yml (Reproducibility)

Conda environments can be defined declaratively:

```yaml
name: fraud-detection
channels:
  - conda-forge
dependencies:
  - python=3.9
  - numpy
  - pandas
```

Create:

```bash
conda env create -f environment.yml
```

Update:

```bash
conda env update -n fraud-detection -f environment.yml
```

This is similar in spirit to a Dockerfile.

---

## What Is `.condarc`?

`.condarc` is Conda’s **configuration file**.

It controls **how Conda behaves**, not what an environment contains.

Comparable to:
- `.npmrc`
- `settings.xml` (Maven)
- `pip.conf`

---

## What `.condarc` Controls

- Channel order and priority
- Auto‑activation behavior
- Environment and package directories
- Proxy and SSL settings
- Solver behavior

It is **global or user‑level policy**, not project configuration.

---

## Recommended `.condarc` Baseline

```yaml
channels:
  - conda-forge
  - defaults

channel_priority: strict
auto_activate_base: false
```

Why this matters:
- `conda-forge` provides consistent modern builds
- `channel_priority: strict` avoids ABI breakage
- Disabling base auto‑activation keeps shells clean

---

## Managing `.condarc`

Preferred via CLI:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda config --set auto_activate_base false
```

Audit configuration:

```bash
conda config --show-sources
conda config --show
```

---

## `.condarc` vs `environment.yml`

| File | Purpose |
|----|----|
| `.condarc` | Global Conda behavior / policy |
| `environment.yml` | Project dependencies |

Typically:
- `.condarc` is **not committed** to repos
- `environment.yml` **is committed**

---

## Mental Model (Dev / ML Friendly)

Think of Conda as:

> A **user‑space runtime manager** providing Docker‑like reproducibility without containers.

- No daemon
- No root access
- Pure directory isolation + shell control

---

## Key Takeaways

- Conda is more than `venv`
- Activation modifies shell state
- Native libraries are the real differentiator
- Use `conda-forge` with strict priority
- Prefer Conda installs, fall back to pip only when needed
- Capture environments with `environment.yml`
- Treat `.condarc` as policy, not project config

---

_End of Conda summary._

