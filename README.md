# Messy Data Generators for Data Cleaning Practice

While learning data science and data cleaning in Python, I noticed that most beginner datasets (like Titanic or Iris) are already way too clean. Real-world data is full of typos, broken dates, weird negative numbers, and missing values hidden as text like `"N/A"` or `"-999"`.

I built this project for myself and other students who want a simple way to generate realistic, messy datasets to practice on. 

Instead of making random tables, I connected the three scripts around a single story: **a graduate statistics student who also works part-time as a data analyst at an e-commerce company and helps out in a university research lab.**

---

## What's Inside

There are three standalone Python scripts. None of them depend on each other, and they only require `pandas` and `numpy`:

1. **`generate_academic_data.py` (University Department Records)**  
   Generates student course records, midterm and final marks, attendance percentages, and teacher ratings.
2. **`generate_ecommerce_data.py` (E-Commerce Order Logs)**  
   Generates online shopping orders with prices, quantities, discounts, delivery fees, and local payment methods (`bKash`, `Nagad`, `Cash on Delivery`).
3. **`generate_research_data.py` (Research Lab Publication Tracker)**  
   Generates academic paper records, submission and publication dates, peer-review rounds, grant status, and citation counts.

---

## How the "Messiness Dial" Works

At the top of each script, there are only three settings you need to touch:

```python
num_rows = 500      # How many rows you want
random_seed = 42    # Change this to get a different random batch
messiness = 0.6     # A dial from 0.0 (completely clean) to 1.0 (very messy)
```

I didn't want `messiness = 1.0` to delete 100% of the data, because then there would be nothing left to clean. So each error type has a realistic maximum limit (`BASE_RATE`), and the probability of an error appearing in a cell is scaled using a simple Bernoulli probability formula:

**P(Error) = Messiness × Base Rate**

| Messiness Level | What Happens to the Data |
| :--- | :--- |
| **`0.0` (Clean)** | No errors at all. Good if you want a clean baseline to compare against. |
| **`0.3` (Light)** | ~5% to 9% errors. Standard `NaN` missing values, some lowercase/uppercase mix-ups, and a few duplicate rows. |
| **`0.6` (Moderate)** | ~15% to 18% errors. Missing values start hiding as strings (`"N/A"`, `""`, `"TBD"`), dates get mixed up (`DD/MM/YYYY` vs `YYYY-MM-DD`), and impossible numbers appear (like negative exam scores or publication dates before submission dates). |
| **`1.0` (Heavy)** | ~30% to 35% max errors. Heavy missing data, numeric placeholder codes (`-999`), impossible calendar dates (`2024-02-30`), extreme outliers, and around 10% duplicate rows. |

---

## Methodology & References

To make sure the error types and percentages are realistic rather than completely random, I designed the error rules around standard data quality frameworks:

* **Error Categories:** Based on the 5 core dimensions from **DAMA-DMBOK2** and **ISO/IEC 25012** (*Completeness, Consistency, Validity, Accuracy, and Uniqueness*).
* **Error Rates (5% to 35% ceiling):** Follows the benchmark noise injection ranges used in database cleaning research papers, specifically **BART** (*Arocena et al., PVLDB 2015*) and **Raha** (*Mahdavi et al., PVLDB 2019*).
* **Missing Data Logic:** Modelled as Missing Completely at Random (**MCAR**) following *Little & Rubin (2019), Statistical Analysis with Missing Data*.

---

## How to Run the Scripts

Make sure you have `pandas` and `numpy` installed:

```bash
pip install pandas numpy
```

Then run any of the scripts in VS Code, terminal, or Google Colab:

```bash
python generate_academic_data.py
```

The script will print a quick summary in the terminal and save a `.csv` file in the same folder so you can load it into a Jupyter Notebook and start cleaning.
