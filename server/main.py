from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    host = "0.0.0.0"
    uvicorn.run(app, 
                host=host, 
                port=8000)