# NaviCube

**NaviCube** is a sophisticated 3D navigation tool designed to enhance spatial orientation in complex datasets. It provides intuitive controls and visualization capabilities for researchers and developers working with multidimensional data.

## Features

- **Interactive 3D Navigation**: Seamlessly explore data in a three-dimensional space with intuitive controls.
- **Customizable Views**: Adjust perspectives and visualization parameters to suit your analysis needs.
- **Integration-Friendly**: Easily integrates with existing data processing pipelines and visualization tools.

## Installation

To incorporate NaviCube into your project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/prostaff4ever/NaviCube.git


## 🔍 Dipole Modeling and Baseline Estimation Pipeline

This module includes a complete workflow for identifying and subtracting magnetic dipole sources from a 3D vector field in order to isolate a global baseline magnetic field.

### Features
- ✅ Full 3D vector moment dipole modeling
- 🎯 Dipole localization using gradient analysis
- 🔧 Parametric fitting of (x, y, z, mₓ, mᵧ, m_z) for each dipole
- ➖ Subtraction of modeled dipole contributions
- 📐 Recovery of global baseline field via low-pass filtering
- 📊 3D visualization of dipole vectors

### Included Files
- `dipole_full_3dm_pipeline.ipynb` – Jupyter notebook with complete annotated workflow
- `dipoletools_3dm.py` – Modular utilities for dipole modeling, fitting, and subtraction

### Quick Start
1. Open the notebook:
   ```bash
   jupyter notebook dipole_full_3dm_pipeline.ipynb
