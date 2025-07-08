# Python Emotiv

Python library to access Emotiv EPOC EEG headset data.

## Python 3 Compatibility

This library has been updated to be fully compatible with Python 3.6+. The following changes were made:

- Added `from __future__ import print_function, division` to all Python files
- Changed all `print "text"` statements to `print("text")` function calls
- Updated exception handling syntax from `except Exception, e:` to `except Exception as e:`
- Changed `raw_input()` to `input()` for user input
- Updated `xrange()` to `range()` (Python 3's range is already efficient)
- Changed integer division `/` to `//` where integer division is needed
- Updated string handling for bytes in socket communications
- Changed `dict.values()` to `list(dict.values())` where needed
- Updated `data.tostring()` to `data.tobytes()` for numpy arrays
- Changed `np.fromstring()` to `np.frombuffer()` for better compatibility

## Installation

### Modern Installation (Recommended)

```bash
# Install using pip with pyproject.toml
pip install .

# Or install in development mode
pip install -e .
```

### Alternative Installation Methods

```bash
# Using setup.py (legacy, may show deprecation warnings)
python setup.py install

# Using pip with requirements
pip install -r requirements.txt
```

## Dependencies

- Python 3.6+
- numpy>=1.19.0
- scipy>=1.5.0
- pyusb>=1.0.0
- pycrypto>=2.6.0 (for encryption)
- matplotlib>=3.0.0 (for plotting examples)
- nitime>=0.7.0 (for spectral analysis)

### Optional Dependencies

For development:
```bash
pip install -e ".[dev]"
```

For documentation:
```bash
pip install -e ".[docs]"
```

## Usage

Basic usage example:

```python
from emotiv import epoc

# Initialize headset
headset = epoc.EPOC()

# Get a sample
data = headset.get_sample()

# Acquire data for 10 seconds
data = headset.acquire_data(10)

# Clean up
headset.disconnect()
```

## Examples

See the `examples/` directory for various usage examples including:
- Basic data recording
- SSVEP (Steady State Visual Evoked Potential) experiments
- BCI (Brain-Computer Interface) applications
- Real-time data streaming with LSL

## Development

### Setting up a development environment

```bash
# Clone the repository
git clone https://github.com/ozancaglayan/python-emotiv.git
cd python-emotiv

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"
```

### Running tests

```bash
pytest
```

## License

This project is licensed under the GNU General Public License v3. See the COPYING file for details.

## Author

Ozan Çağlayan <ozancag@gmail.com>
