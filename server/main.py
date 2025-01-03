from fastapi import FastAPI
from api import model_router
from fastapi.responses import RedirectResponse
import uvicorn

app = FastAPI()

# Set Router
app.include_router(model_router.router, prefix="/model", tags=['model'])

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "Healthy server"}

if __name__ == "__main__":
    host = "0.0.0.0"
    uvicorn.run("main:app", 
                host=host, 
                port=8000,
                reload=True)