from main import app
import uvicorn
import os


def main():
    port = int(os.environ.get("PORT", 7860))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()
