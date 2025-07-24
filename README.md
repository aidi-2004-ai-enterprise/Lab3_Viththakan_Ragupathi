# 🐧 Lab 3 – Penguins Species Classification with XGBoost & FastAPI

**Author:** Viththakan Ragupathi  
**Course:** AIDI‑2004 – AI Enterprise  
**Assignment:** Lab 3  
**Goal:** Build and deploy a machine learning model to classify penguin species.

---

## 🧠 Project Description

This project involves training an **XGBoost** classifier using the [Palmer Penguins dataset](https://github.com/allisonhorst/palmerpenguins) and deploying it as a web API using **FastAPI**. The API accepts penguin measurements as input and returns the predicted species.

It includes:

- Clean preprocessing (one-hot for categorical fields, label encoding for target)
- XGBoost classifier with anti-overfitting parameters
- RESTful API endpoint (`/predict`) with input validation
- Interactive Swagger UI at `/docs`
- Model and encoder saved for reuse during prediction

---

## 🗂️ Project Structure
```bash
Lab3_Viththakan_Ragupathi/
├── train.py # Train model and save artifacts
├── app/
│ ├── main.py # FastAPI app
│ └── data/
│ ├── model.json # Saved XGBoost model
│ └── preprocess_meta.json # Stores encoders & label mappings
├── requirements.txt
└── README.md # This file
```



## 🚀 Getting Started

### 1. Install Dependencies

First, create a virtual environment and activate it:

```bash
python -m venv .venv
.venv\Scripts\activate   # (for Windows)
Then install the required libraries:
```
Then install the required libraries:

```bash
pip install -r requirements.txt
```
2. Train the Model
   
Run the training script:
```bash
python train.py
```
This will generate:

model.json: the trained model

preprocess_meta.json: encoders and label mappings used during training

3. Start the API Server
```bash
uvicorn app.main:app --reload
```
Open your browser and visit: http://127.0.0.1:8000/docs to access the Swagger UI.

🎯 API Endpoint
POST /predict
Predicts the species of a penguin based on input features.

✅ Sample Request Body
```bash
{
  "bill_length_mm": 44.0,
  "bill_depth_mm": 17.5,
  "flipper_length_mm": 190,
  "body_mass_g": 4200,
  "year": 2008,
  "sex": "female",
  "island": "Torgersen"
}
```
✅ Sample Response
```bash
{
  "species": "Adelie"
}
```
❌ Error Handling
Invalid Category
If an invalid value is sent for an enum field (e.g., island: Atlantis):

```bash
{
  "detail": [
    {
      "loc": ["body", "island"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum"
    }
  ]
}
```
Missing Field
If a required field is missing:
```bash
{
  "detail": [
    {
      "loc": ["body", "sex"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
📦 Dependencies
Make sure your requirements.txt includes at least:

1. fastapi
2. uvicorn
3. scikit-learn
4. pandas
5. xgboost
6. pydantic

📚 Dataset Info
The Palmer Penguins dataset includes measurements for three species:

1. Adelie
2. Chinstrap
3. Gentoo

Features used in the model:

1. bill_length_mm
2. bill_depth_mm
3. flipper_length_mm
4. body_mass_g
5. year
6. sex
7. island

