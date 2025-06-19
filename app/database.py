"""
Database connection and session management.
Implements async database operations with proper connection pooling.
"""
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from config import get_settings

settings = get_settings()

# Create async engine with connection pooling
engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    **settings.database_config
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with proper cleanup."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with get_db_session() as session:
        yield session


class DatabaseRepository:
    """Base repository pattern for database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, model_instance):
        """Create new record."""
        self.session.add(model_instance)
        await self.session.flush()
        return model_instance
    
    async def get_by_id(self, model_class, record_id):
        """Get record by ID."""
        return await self.session.get(model_class, record_id)
    
    async def delete(self, model_instance):
        """Delete record."""
        await self.session.delete(model_instance)
        await self.session.flush()
    
    async def bulk_insert(self, model_instances):
        """Bulk insert records for better performance."""
        self.session.add_all(model_instances)
        await self.session.flush()
        return model_instances
