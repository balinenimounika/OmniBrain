from fastapi import FastAPI

app = FastAPI(title="OmniBrain API")


@app.get("/")
def home():
    return {"message": "OmniBrain API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}