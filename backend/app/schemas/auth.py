from pydantic import BaseModel, ConfigDict, field_validator


class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        # Lowercase and strip whitespace before anything touches it
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def check_password_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.strip().lower()


class UserResponse(BaseModel):
    # Allows Pydantic to read values directly from SQLAlchemy ORM objects
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: str
    plan: str
    # Computed from the ADMIN_EMAILS allow-list (User.is_admin property). Used by
    # the frontend to show the admin nav item; access is enforced server-side.
    is_admin: bool = False


class AuthResponse(BaseModel):
    user: UserResponse
    token: str