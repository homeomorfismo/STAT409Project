# DNNPype - DNN Modeling for Acoustic Pipes

DNN modeling for acoustic pipes using Flax/JAX. This project provides tools for acoustic pipe modeling, unit conversion, and audio sample classification.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Git Setup](#git-setup)
- [Getting the Code](#getting-the-code)
- [Package Installation](#package-installation)
- [Usage](#usage)
- [Development](#development)

## Prerequisites

### Python Installation

You need Python 3.10 or higher installed on your system.

#### Windows
- Download from [python.org](https://www.python.org/downloads/windows/)
- Or install via Microsoft Store
- **Detailed guide**: [Python Windows Installation Guide](https://docs.python.org/3/using/windows.html)

#### macOS
- Download from [python.org](https://www.python.org/downloads/macos/)
- Or use Homebrew: `brew install python`
- **Detailed guide**: [Python macOS Installation Guide](https://docs.python.org/3/using/mac.html)

#### Linux
- Ubuntu/Debian: `sudo apt update && sudo apt install python3 python3-pip python3-venv`
- Fedora: `sudo dnf install python3 python3-pip`
- Arch: `sudo pacman -S python python-pip`
- **Detailed guide**: [Python Linux Installation Guide](https://docs.python.org/3/using/unix.html)

## Installation

### Step 1: Create a Virtual Environment

A virtual environment isolates your project dependencies from your system Python.

```bash
# Create virtual environment
python -m venv dnnpype-env

# Alternative name if you prefer
python3 -m venv dnnpype-env
```

### Step 2: Activate the Virtual Environment

#### Windows (Command Prompt)
```cmd
dnnpype-env\Scripts\activate
```

#### Windows (PowerShell)
```powershell
dnnpype-env\Scripts\Activate.ps1
```

#### macOS/Linux
```bash
source dnnpype-env/bin/activate
```

**Note**: You'll see `(dnnpype-env)` in your terminal prompt when the environment is active.

### Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

## Git Setup

### Install Git
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **macOS**: `brew install git` or download from [git-scm.com](https://git-scm.com/download/mac)
- **Linux**: `sudo apt install git` (Ubuntu/Debian) or equivalent for your distro

### Configure Git
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Complete Git tutorial**: [Git Handbook](https://guides.github.com/introduction/git-handbook/) or [Atlassian Git Tutorial](https://www.atlassian.com/git/tutorials)

## Getting the Code

### Clone the Repository
```bash
git clone https://github.com/homeomorfismo/STAT409Project.git
cd STAT409Project
```

## Package Installation

### Install Dependencies and Package
```bash
# Install the package in editable mode with development dependencies
pip install -e ".[dev]"
```

This will install:
- All required dependencies (numpy, jax, flax, optax, sounddevice, rich, scipy)
- Development tools (pytest, pytest-cov)
- The package itself in editable mode

### Verify Installation
```bash
# Check if the package is installed
pip list | grep dnnpype

# Test imports
python -c "import dnnpype; print('DNNPype installed successfully!')"
```

## Usage

Once installed, you have access to three command-line tools:

### 1. Unit Converter (`convert_units`)

Convert between different pressure units.

#### Basic Usage
```bash
convert_units --help
```

#### Example Usage
```bash
# Convert 10 pascals to mmH2O
convert_units --value 10 --input pascal --output mmH2O
# Output: From 10.0 pascal to 1.01974428892211 mmH2O

# Convert 100 mmHg to pascal
convert_units --value 100 --input mmHg --output pascal

# Convert 1 atm to psi
convert_units --value 1 --input atm --output psi
```

### 2. Audio Sample Classifier (`classify_samples`)

Generate and classify audio samples for acoustic analysis.

#### Basic Usage
```bash
classify_samples --help
```

#### Example Usage
```bash
# Generate samples at 440Hz and save to output directory
classify_samples --frequency 440 --output-dir ./audio_samples

# Generate with custom parameters
classify_samples \
    --frequency 440 \
    --output-dir ./audio_samples \
    --duration 2.0 \
    --samplerate 44100 \
    --samples 10 \
    --save-samples \
    --plot-samples
```

#### Parameters
- `--frequency`: Frequency in Hz (required)
- `--output-dir`: Directory to save outputs (required)
- `--duration`: Duration in seconds (default: 1.0)
- `--samplerate`: Sample rate (default: 44100)
- `--samples`: Number of samples to generate (default: 5)
- `--output`: Output file name
- `--save-samples`: Save audio samples to files
- `--plot-samples`: Generate plots of the samples

### 3. DNN Model Runner (`run_model`)

Train or evaluate DNN models for organ pipe acoustics.

#### Basic Usage
```bash
run_model --help
```

#### Training Examples
```bash
# Basic training
run_model --mode train --data_path ./Data/allOrgan.csv

# Training with custom parameters
run_model \
    --mode train \
    --epochs 300 \
    --learning_rate 0.01 \
    --batch_size 64 \
    --data_path ./Data/allOrgan.csv \
    --rng_seed 42

# Quick training for testing
run_model \
    --learning_rate 0.01 \
    --epochs 300 \
    --data_path ../Data/allOrgan.csv
```

#### Evaluation Examples
```bash
# Evaluate trained model
run_model \
    --mode evaluate \
    --data_path ./Data/allOrgan.csv \
    --batch_size 32
```

#### Parameters
- `--mode`: Operation mode (`train` or `evaluate`, default: `train`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--data_path`: Path to CSV data file
- `--rng_seed`: Random seed for reproducibility

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dnnpype

# Run specific test file
pytest tests/test_specific.py
```

### Code Formatting and Linting
```bash
# Format code with Black
black src/

# Lint with Ruff
ruff check src/

# Fix linting issues automatically
ruff check src/ --fix
```

### Deactivating Virtual Environment
When you're done working:
```bash
deactivate
```

## Project Structure

```
STAT409Project/
├── LICENSE
├── pyproject.toml
├── README.md
└── src/
    └── dnnpype/
        ├── __init__.py
        ├── appSound.py      # Audio sample classification
        ├── convert.py       # Unit conversion utilities
        ├── dnn.py          # DNN model implementation
        ├── sound.py        # Sound processing utilities
        └── testCode.py     # Test implementations
```

## Troubleshooting

### Common Issues

1. **"Command not found" errors**: Make sure your virtual environment is activated
2. **Import errors**: Verify the package is installed with `pip list | grep dnnpype`
3. **Permission errors**: On Unix systems, you might need to use `python3` instead of `python`

### Getting Help

1. Check the help for any command: `<command> --help`
2. Verify Python version: `python --version` (should be 3.10+)
3. Check virtual environment: Look for `(dnnpype-env)` in your prompt

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## License

This project is licensed under the MPL-2.0 License - see the [LICENSE](../LICENSE) file for details.

## Author

Gabriel Pinochet-Soto (gpin2@pdx.edu)
