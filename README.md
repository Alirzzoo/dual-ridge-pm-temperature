# Dual-Scenario RBF-Ridge for Rotor-Magnet Temperature

### Hybrid Thermal-Aware Regression for PMSM Rotor Temperature Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-F59E0B?style=for-the-badge)]()

**Official implementation accompanying:**

> **Rotor Magnet Temperature Estimation in Permanent Magnet Motors Using Ridge Regression**
> Alireza Rezaei — *2026*

[Overview](#-overview) • [Installation](#-installation) • [Data](#-data) • [Quick Start](#-quick-start) • [Method](#-method) • [Results](#-results) • [Citation](#-citation)

</div>

---

## ⚠️ Scope and Limitations

<div align="center">

| ✅ What this repository **is** | ❌ What it **is not** |
|:---|:---|
| A dual-scenario Ridge-based PM temperature estimator | A universal motor-temperature model |
| A reproducible 3-fold GroupKFold study | A claim of state-of-the-art performance |
| A hybrid RBF + thermal-history regression framework | A substitute for physical thermal modeling |
| A research prototype | A validated deployment-ready tool |

</div>

> **Important:** Results are specific to the publicly available *Electric Motor Temperature* dataset and the exact preprocessing, feature engineering, and hyperparameter search implemented here. **No independent external test set was evaluated.** The out-of-fold metrics describe fold-trained models, not the final model fitted on all development data.

---

## 📖 Overview

This repository implements a **dual-scenario** rotor-magnet (PM) temperature estimator for permanent magnet synchronous motors (PMSMs):

- **Operational** (8 inputs): `u_d`, `u_q`, `i_d`, `i_q`, `motor_speed`, `torque`, `coolant`, `ambient`
- **Sensor-assisted** (11 inputs): adds `stator_winding`, `stator_tooth`, `stator_yoke`

### Architecture

For the **operational** scenario, a **soft hybrid** combines:

1. an **RBF-Ridge** model on current/speed history features
2. a **Thermal-Ridge** model that additionally uses thermal-history features

The blend weight decays with profile age:

$$
w(\text{age}) = \exp\left(-\frac{\text{age}}{\tau}\right)
$$

where $\tau \in \{30, 300, 1800\}$ seconds is selected on inner validation. For the **sensor-assisted** scenario, only the RBF-Ridge model is used.

---

## 🚀 Installation

### Prerequisites

- Python **3.10** or **3.11**
- ~4 GB RAM (for the largest inner-grid search)
- CPU is sufficient (no GPU required)

### Setup

```bash
git clone https://github.com/Alirzzoo/dual-ridge-pm-temperature.git
cd dual-ridge-pm-temperature

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
