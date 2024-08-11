import uvicorn
import app.routers.main


if __name__ == "__main__":
    uvicorn.run(app.routers.main.app, host="0.0.0.0", port=8000)