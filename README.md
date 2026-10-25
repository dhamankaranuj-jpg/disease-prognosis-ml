# Disease Prognosis using Machine Learning

Predicts a disease outcome from patient data using four classifiers (**Random Forest, XGBoost, Decision Tree, Logistic Regression**) and compares them on five parameters: Accuracy, Precision, Recall, F1-Score and AUC-ROC.

Built and tested on the UCI Chronic Kidney Disease dataset (400 patients, 24 clinical attributes), and works with any CSV / Excel / ZIP dataset that has a result column.

## Try it
| | Link |
|---|---|
| Web app | _paste your Streamlit link here_ |
| Notebook (Google Colab) | [Open in Colab](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/Disease_Prognosis_Colab.ipynb) |

## What it does
- Cleans the data automatically (missing values, spaces, typing errors, ID columns)
- Splits it 80% training / 20% testing and trains four models
- Compares the models and picks the best one
- Predicts a single patient from a form, and scores all patients with a downloadable CSV
- Shows which attributes drive the prediction

## Results on the CKD sample (20% unseen test data)
| Model | Accuracy |
|---|---|
| Logistic Regression | 98.75% |
| Random Forest | 97.50% |
| Decision Tree | 97.50% |
| XGBoost | 96.25% |

## Files
| File | Purpose |
|---|---|
| `disease_prognosis_app.py` | Streamlit web app |
| `Disease_Prognosis_Colab.ipynb` | Step-by-step notebook for Google Colab |
| `sample_data/kidney_disease.csv` | Sample CKD dataset, shown when nothing is uploaded |
| `requirements.txt` | Python libraries needed |

## Run on your computer
```bash
pip install -r requirements.txt
python -m streamlit run disease_prognosis_app.py
```

## Note
Student project for learning. It is not a medical device and must not be used for real diagnosis.
