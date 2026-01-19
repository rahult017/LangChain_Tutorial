from fastapi import FastAPI
import uvicorn

app =FastAPI()

@app.get("/reverse")
def reverse_text(text:str):
    return {"orinal": text,"rev":text[::-1]}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5000,
        reload=True
    )
#uvicorn main:app --host 127.0.0.1 --port 5000 --reload 