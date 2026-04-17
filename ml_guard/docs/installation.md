# Installation & Setup

## Prerequisites
- Node.js v18+
- Python 3.10+
- PostgreSQL or SQLite
- Redis (for Celery background tasks)

## Backend Setup (FastAPI)
1. Navigate to the backend directory: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Setup `.env` using `.env.example` as a template.
4. Apply database migrations: `alembic upgrade head`
5. Start development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
6. Start Celery Worker (In a separate terminal):
   ```bash
   celery -A app.worker worker --loglevel=info
   ```

## Frontend Setup (Next.js)
1. Navigate to frontend directory: `cd frontend`
2. Install dependencies: `npm install`
3. Start the Next SDK dev server:
   ```bash
   npm run dev
   ```

You can alternatively run `start_services.bat` on Windows to launch all 3 microservices at once!
