# RAG Chatbot

A React and FastAPI personal-profile chatbot powered by LangChain, sentence-transformer embeddings, and a local LLM.

## Run locally

Start the API:

```powershell
.\virtual_env\Scripts\uvicorn.exe main:app --reload
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Environment variables

The frontend uses `VITE_API_URL` to find the API. It defaults to `http://127.0.0.1:8000` for local development.
