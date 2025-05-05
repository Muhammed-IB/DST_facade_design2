# 🏢 AHP-Based Façade Design Decision Support Tool

This interactive tool ranks façade design alternatives using the Analytic Hierarchy Process (AHP), incorporating multiple performance metrics and user-defined weights. It supports real-time ranking, visualization, and consistency checks — powered by Streamlit, Google Sheets, and `ahpy`.

---

## 📚 Table of Contents

* [Why This Exists](#-why-this-exists)
* [Process Overview](#-process-overview)
* [How It Works ](#-how-it-works)
* [Repository Layout](#-repository-layout)
* [Quick Start](#-quick-start)
* [Detailed Workflow](#-detailed-workflow)
* [Implementation Decisions](#-implementation-decisions)
* [Limitations & Future Work](#-limitations--future-work)
* [References](#-references)

---

## ✨ Why This Exists

Early-stage façade design often requires balancing multiple performance and cost-related factors. This tool enables:

* Transparent decision-making with AHP
* Real-time sensitivity tuning
* Visualization of trade-offs through parallel coordinate plots and ranking tables
* Integration with Google Sheets for collaborative CSV data updates

---

## 🚦 Process Overview

📊 CSV Inputs → 🧮 Normalize Data → 🪼 Define Weights → 🔄 AHP Ranking → 🖼️ Visual Output

---

## ⚙️ How It Works (TL;DR)

* CSVs are loaded from Google Sheets (one per metric)
* Sliders in the UI allow users to adjust weights for criteria and sub-criteria
* The `ahpy` library builds the AHP comparison tree dynamically
* Results are ranked and visualized as:

  * Tables of alternatives
  * Image grid
  * Parallel coordinate plot

---

## 📁 Repository Layout

```
.
├─ AHP_project.py           # Streamlit app script
├─ requirements.txt         # Python dependencies
├─ images/                  # PNGs of façade design alternatives
├─ DSC_CaseStudy2_Output/   # Output images for Pareto/Elitist fronts
├─ JoyceStreet_data.csv     # Input dataset used for NSGA-II
├─ pareto_front.csv         # Auto-generated Pareto front
├─ elitist_front_2-6.csv    # Elitist fronts from optimization
└─ README.md
```

---

## ⚡ Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

2. **Install requirements**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run AHP_project.py
```

---

## 🔍 Detailed Workflow

| Step | Script/File           | Purpose                                |
| ---- | --------------------- | -------------------------------------- |
| 1    | AHP\_project.py       | UI + ranking logic                     |
| 2    | JoyceStreet\_data.csv | Provides raw design data for MOO       |
| 3    | pymoo (NSGA-II)       | Generates Pareto + elitist fronts      |
| 4    | ahpy                  | Builds AHP model & scores alternatives |
| 5    | Streamlit + Plotly    | Provides interactive sliders & plots   |

---

## 🧽 Workflow Diagram

![Asset 8](https://github.com/user-attachments/assets/3f46ceae-55c7-4a3a-9d51-546f01ea23cb)


---

## 🧠 Implementation Decisions

| Decision                            | Rationale                                    |
| ----------------------------------- | -------------------------------------------- |
| Use `ahpy` for pairwise ranking     | Lightweight, supports full hierarchy         |
| Use `pymoo` for Pareto optimization | Well-supported multi-objective algorithms    |
| Allow AHP weights via sliders       | Easy UX for iterative design trade-offs      |
| Store fronts in CSV                 | Allows later use without recomputation       |
| Visualize with Matplotlib & Plotly  | Combines static plots and interactive graphs |

---

## ⚠️ Limitations & Future Work

### Current Limitations

* No user upload of CSVs yet
* NSGA-II is recomputed only if output files are missing
* Fixed number of elitist fronts shown (up to 6)
* No interactive consistency checks for AHP inputs

### Future Enhancements

* Allow custom uploads of front CSVs
* Add comparison with baseline design
* Integrate export as PDF report
* Add filters for design constraints (e.g., max cost or UDI threshold)

---

## 📚 References

* [`ahpy` documentation](https://python-ahpy.readthedocs.io/en/latest/)
* [`pymoo` optimization docs](https://pymoo.org/)
* [Streamlit Docs](https://docs.streamlit.io/)
* Saaty, T.L. (1980). *The Analytic Hierarchy Process*.
* buildingSMART (2020). *IFC 4.3 Final* Specification

---

👤 Created by Mohammed Hassen
🗓️ Spring 2025 | Georgia Tech | ARCH 8833

