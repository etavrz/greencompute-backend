from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column, DateTime, Integer, Text
from sqlalchemy.orm import DeclarativeBase, mapped_column

EMBEDDINGS_DIM = 768  # dimensions for all-mpnet-base embeddings


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(BigInteger, primary_key=True)
    embeddings = Column(Vector(EMBEDDINGS_DIM), nullable=True, name=f"embeddings@{EMBEDDINGS_DIM}")
    doc_title = Column(Text)
    url = Column(Text, nullable=True)
    content = Column(Text)
    tokens = Column(Integer)
    date_indexed = mapped_column(DateTime)
