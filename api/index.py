
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io, re, os, smtplib
from email.message import EmailMessage

app = FastAPI()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

def send_email(receiver_email, file_bytes):
    msg = EmailMessage()
    msg["Subject"] = "Your TOPSIS Result"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg.set_content("Your TOPSIS result is attached.")

    msg.add_attachment(file_bytes, maintype="application", subtype="octet-stream", filename="result.csv")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)

@app.post("/api/topsis")
async def run_topsis(
    file: UploadFile = File(...),
    weights: str = Form(...),
    impacts: str = Form(...),
    email: str = Form(...)
):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return JSONResponse(content={"error": "Invalid email format"}, status_code=400)

    try:
        weights = list(map(float, weights.split(",")))
        impacts = impacts.split(",")
    except:
        return JSONResponse(content={"error": "Invalid weights or impacts format"}, status_code=400)

    if len(weights) != len(impacts):
        return JSONResponse(content={"error": "Weights and impacts count mismatch"}, status_code=400)

    for i in impacts:
        if i not in ["+", "-"]:
            return JSONResponse(content={"error": "Impacts must be + or -"}, status_code=400)

    content = await file.read()

    try:
        df = pd.read_csv(io.BytesIO(content))
    except:
        return JSONResponse(content={"error": "Invalid CSV file"}, status_code=400)

    if df.shape[1] < 3:
        return JSONResponse(content={"error": "CSV must have at least 3 columns"}, status_code=400)

    data = df.iloc[:, 1:].values.astype(float)

    norm = np.sqrt((data ** 2).sum(axis=0))
    data = data / norm

    for i in range(len(weights)):
        data[:, i] *= weights[i]

    ideal_best = []
    ideal_worst = []

    for i in range(len(impacts)):
        if impacts[i] == "+":
            ideal_best.append(data[:, i].max())
            ideal_worst.append(data[:, i].min())
        else:
            ideal_best.append(data[:, i].min())
            ideal_worst.append(data[:, i].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    dist_best = np.sqrt(((data - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((data - ideal_worst) ** 2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)
    rank = score.argsort()[::-1] + 1

    df["Topsis Score"] = score
    df["Rank"] = rank

    output = io.StringIO()
    df.to_csv(output, index=False)
    output_bytes = output.getvalue().encode()

    try:
        send_email(email, output_bytes)
    except Exception as e:
        return JSONResponse(content={"error": "Failed to send email", "details": str(e)}, status_code=500)

    return {"message": "TOPSIS result sent to your email successfully"}
