import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

user = os.getenv("POSTGRES_USER")
pwd = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
db = os.getenv("POSTGRES_DB")

try:
    engine = create_engine(f"postgresql+psycopg2://{user}:{pwd}@{host}/{db}")
except Exception as e:
    raise Exception(f"Could not connect to Postgres: {e}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
