# Disease Prognosis using Machine Learning

A web app that predicts a disease outcome from patient data. Upload any dataset and it trains four classifiers, compares them, and shows live results.

**Models:** Random Forest, XGBoost, Decision Tree, Logistic Regression
**Parameters checked:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

## Try it
| | Link |
|---|---|
| Web app | _paste your Streamlit link here_ |
| Notebook (Google Colab) | [Open in Colab](https://colab.research.google.com/github/dhamankaranuj-jpg/disease-prognosis-ml/blob/main/Disease_Prognosis_Colab.ipynb) |

## How to use the web app
1. Open the web app link.
2. Upload a dataset (`.csv`, `.xlsx` or `.zip`) from the left panel.
3. Pick the result column if it was not detected automatically.
4. The four models are trained and compared. Results appear live for the uploaded dataset.

## What it does
- Cleans the data automatically (missing values, spaces, typing errors, ID columns)
- Splits it 80% training / 20% testing and trains four models
- Compares the models and picks the best one
- Predicts a single patient from a form
- Scores all patients and lets you download the result as CSV
- Shows which attributes drive the prediction

## Limits
Up to 5000 rows, 100 columns and 20 result classes per dataset.

## Files
| File | Purpose |
|---|---|
| `disease_prognosis_app.py` | Streamlit web app |
| `Disease_Prognosis_Colab.ipynb` | Step-by-step notebook for Google Colab |
| `requirements.txt` | Python libraries needed |

## Run on your computer
```bash
pip install -r requirements.txt
python -m streamlit run disease_prognosis_app.py
```

## Note
Student project for learning. It is not a medical device and must not be used for real diagnosis.
