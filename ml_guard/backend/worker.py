from app.core.celery_app import celery_app as app

if __name__ == "__main__":
    app.start()
