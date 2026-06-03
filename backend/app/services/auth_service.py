import asyncio
import secrets
from functools import partial

from fastapi import HTTPException, status
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import User, UserToken

# The hashing context, bcrypt is the only scheme in use
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def _hash_password(plain: str) -> str:
    # bcrypt is intentionally slow (100-300ms).
    # run_in_executor moves it off the async event loop
    # so other requests are not blocked while it runs.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _pwd.hash, plain)


async def _verify_password(plain: str, hashed: str) -> bool:
    # Same reason as _hash_password, verify is equally slow.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(_pwd.verify, plain, hashed))


def _new_token() -> str:
    # 32 random bytes → 64-character hex string
    return secrets.token_hex(32)


async def register_user(
    db: AsyncSession, email: str, name: str, password: str
) -> tuple[User, str]:
    # Check for duplicate email at the application level first.
    # The UNIQUE constraint on the column is a second safety net.
    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(
        email=email,
        name=name,
        hashed_password=await _hash_password(password),
    )
    db.add(user)

    # flush() writes the user row so user.id is available,
    # but does not commit yet, both rows commit together below.
    await db.flush()

    token_value = _new_token()
    db.add(UserToken(user_id=user.id, token=token_value))

    await db.commit()
    await db.refresh(user)
    return user, token_value


async def login_user(
    db: AsyncSession, email: str, password: str
) -> tuple[User, str]:
    # selectinload(User.tokens) pre-loads the tokens collection in the same
    # round-trip.  Without it, db.add(UserToken(...)) later causes SQLAlchemy
    # to backpopulate user.tokens, which triggers an implicit lazy SQL query.
    # Implicit SQL in async context raises MissingGreenlet → 500.
    result = await db.execute(
        select(User)
        .where(User.email == email)
        .options(selectinload(User.tokens))
    )
    user = result.scalar_one_or_none()

    # Same message for wrong email AND wrong password on purpose -
    # different messages would let an attacker enumerate registered emails.
    # The try/except ensures a corrupted hash in the DB still returns 401,
    # never an unhandled 500.
    try:
        password_ok = user is not None and await _verify_password(
            password, user.hashed_password
        )
    except Exception:
        password_ok = False

    if user is None or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token_value = _new_token()
    db.add(UserToken(user_id=user.id, token=token_value))
    await db.commit()
    await db.refresh(user)   # mirror register_user, keeps ORM state clean
    return user, token_value


async def get_user_by_token(db: AsyncSession, token: str) -> User | None:
    result = await db.execute(
        select(User)
        .join(UserToken, UserToken.user_id == User.id)
        .where(UserToken.token == token)
    )
    return result.scalar_one_or_none()


async def logout_user(db: AsyncSession, token: str) -> None:
    result = await db.execute(select(UserToken).where(UserToken.token == token))
    row = result.scalar_one_or_none()
    if row:
        await db.delete(row)
        await db.commit()
