import datetime
from sqlalchemy import String, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

# One table in the database for each case model
class Case(Base):
    __tablename__ = "cases"
    
    # Primary key for each case row
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    
    # Short case title 
    title: Mapped[str] = mapped_column(String(200))
    
    # Optional longer description
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Current case status
    status: Mapped[str] = mapped_column(String(50), default="open")

    # Timestamp created by the database automatically
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )