# DataFactory

This project preserves a layered web architecture with `api/service/dao/entity/resources`.

Currently retained:
- Local SQLite initialization (SQL migrations run automatically at startup)
- User authentication and session flow (`/api/auth/login`, `/api/auth/logout`, `/api/auth/session`)
- User management APIs (`/api/users`)

Removed:
- Multi-agent features
- Other business features such as chat, admin, and database probing/execution

## Run

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m web.app
```

After startup:
- Health: `http://127.0.0.1:8888/health`
- Swagger: `http://127.0.0.1:8888/docs`

## Optional: Login Bypass (Auto Admin)

In `/src/web/resources/config.yaml`, set:

```yaml
auth:
  auto_login_as_user: true
```

When enabled, guarded APIs can be called without login. The backend will automatically authenticate using `admin/admin`.
