# Campus Facility Reservation Monitoring — Streamlit

This project adapts the submitted midterm notebook for the Section A scenario in `Final1.pdf`: campus facility reservation monitoring and prediction of cancellation/no-show risk.

## Features
- Home page and navigation
- CSV upload, required-column validation, data preview, missing values, duplicates
- Exploratory statistics and interactive charts
- Prediction with a saved Decision Tree preprocessing/model pipeline
- Decision-support dashboard with KPIs and a review queue
- SAD functional/non-functional requirements and analytical workflow

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `python -m pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Open the local URL shown in the terminal (usually http://localhost:8501).

The app includes the same 15-row synthetic reservation dataset defined in the midterm notebook, so it can open without uploading a CSV. You can upload another CSV from the Dataset page; it must include `Facility_Type`, `User_Category`, `Lead_Time_Days`, `Participants`, and `Usage_Status`.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload all files/folders in this project.
2. Go to https://share.streamlit.io and sign in.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Click **Deploy** and wait for the build to finish.
5. Open the app URL and test each page.

## Model continuity and limitation
The model uses the midterm notebook's target `Reservation_Outcome` (`Completed` / `Not Completed`), predictor columns `Facility_Type`, `User_Category`, `Lead_Time_Days`, and `Participants`, and Decision Tree pipeline approach. The included fitted model was generated from the notebook's built-in synthetic sample so the app is runnable. Confirm its metrics and sample predictions against your exact executed midterm notebook before submitting. The sample has only 15 original rows and only two non-completed outcomes; the model is educational and not reliable enough for real operational decisions.

Do not upload sensitive student data or use model predictions as automatic rejection decisions.
