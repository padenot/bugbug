# Review Helper Backend

FastAPI backend for Mozilla's Review Helper supporting both Phabricator and GitHub platforms.

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Copy environment variables:

   ```bash
   cp .env.example .env
   ```

3. Run database migrations:

   ```bash
   uv run alembic upgrade head
   ```

4. Start the server:
   ```bash
   uv run uvicorn app.main:app --reload
   ```

## API Documentation

Once the server is running, visit `/docs` for interactive API documentation.

## Review logs

Logs emitted while handling a review include the available `review_request_id`,
`revision_id`, and `diff_id`, including logs from libraries such as HTTPX and
Uvicorn. The identifiers follow concurrent tasks independently and remain set
through publication and request cleanup. Startup and health logs have no review
identifiers.

For example:

```text
[review_request_id=3457 revision_id=334842 diff_id=1410580] HTTP Request: GET ...
```

In Cloud Logging, filter the service's logs with
`textPayload=~"review_request_id=3457([^0-9]|$)"` to match the exact request ID.
The revision and diff identifiers become available after the review
request is loaded; the processing URL supplies the request ID before that lookup.

## Cloud SQL Setup

This application supports Google Cloud SQL for PostgreSQL with two connection methods:

### Local Development with Cloud SQL Auth Proxy

1. Install and run the Cloud SQL Auth Proxy:

   Follow instructions from the [Cloud SQL Auth Proxy documentation](https://docs.cloud.google.com/sql/docs/postgres/connect-auth-proxy#install).

2. Configure your `.env`:

   ```bash
   DATABASE_URL=postgresql+asyncpg://user:password@127.0.0.1:5432/reviewhelper
   ```

### Cloud Run Deployment

Cloud Run connects to Cloud SQL via Unix socket automatically. Set these environment variables:

```bash
CLOUD_SQL_INSTANCE=project-id:us-central1:instance-name
DB_USER=reviewhelper
DB_PASS=your-db-password
DB_NAME=reviewhelper
```

Ensure your Cloud Run service has the Cloud SQL connection configured in the deployment settings.

Pushing to the `deploy/reviewhelper-api-dev` branch deploys to the dev environment.

## GitHub review context

Set `GITHUB_TOKEN` in the worker process environment to a GitHub personal access
token for authenticated base-commit checks and landing-merge searches. The
platform tools read it directly from the environment, like `PHABRICATOR_API_KEY`.
A local `.env` must be exported into the process environment for these tools.
Configure the token on both development and production workers.
