# Deploying ml-service to Render

## 1. Push this folder as its own git repo
```bash
cd ml-service
git init && git add . && git commit -m "Initial commit"
git remote add origin <your-ml-service-repo-url>
git push -u origin main
```
(`weights/` is meant to hold your trained `.pt` file — either commit it if
it's under GitHub's 100MB file limit, use Git LFS if it's larger, or skip
committing it and start Render on `DETECTOR_MODE=mock` until you have one.)

## 2. Create the Render service
- Render dashboard → **New → Web Service** → connect this repo.
- **Runtime**: Docker (Render auto-detects the `Dockerfile`).
- **Instance type**: Free tier works for `DETECTOR_MODE=mock`. For
  `DETECTOR_MODE=real`, free tier's 512MB RAM is genuinely tight for
  ultralytics/torch — expect slow cold starts and possible OOM under load;
  upgrade if it becomes a problem, don't assume free tier will hold up in
  production with real inference.
- Render sets `PORT` automatically — the Dockerfile already binds to it
  (`--port ${PORT:-8000}`), nothing to configure there.
- torch/ultralytics are NOT installed by default (see `requirements-real.txt`)
  to keep builds fast when you're still on `DETECTOR_MODE=mock`. When you're
  ready for `DETECTOR_MODE=real`, set the Docker build argument
  `INSTALL_REAL_DETECTOR=true` in Render's dashboard (Settings → Build →
  Docker Build Args, or via a `render.yaml` if you use one) — otherwise the
  service will start but crash on the first scan when it tries to import
  `ultralytics` and finds it isn't installed.

## 3. Environment variables
Set every variable from `.env.example` in the Render dashboard's
**Environment** tab (Render doesn't read `.env` files from the repo —
they need to be entered there directly):

| Variable | Value |
|---|---|
| `DATABASE_URL` | your main Neon connection string |
| `TRAINING_DB_URLS` | your 3-4 training Neon connection strings, comma-separated |
| `DETECTOR_MODE` | `mock` to start, `real` once you have weights |
| `CLOUDINARY_CLOUD_NAME` / `CLOUDINARY_API_KEY` / `CLOUDINARY_API_SECRET` | from your Cloudinary dashboard |
| `CORS_ALLOW_ORIGINS` | `["https://<your-vercel-domain>"]` — **update this once you know your Vercel URL**, or the UI's direct calls (add-food, feedback) will be blocked by CORS |

## 4. Deploy and verify
Render builds and deploys automatically on push. Once live:
```bash
curl https://<your-ml-service>.onrender.com/health
```
should return `{"status": "ok", ...}`.

## Note on free-tier cold starts
Render's free web services spin down after ~15 minutes of inactivity and
take 30-60s to wake on the next request. The first scan/search after idle
time will be slow — this is Render's behavior, not a bug in the app. If
that's unacceptable for a demo, either upgrade the instance or hit the
`/health` endpoint on a schedule (e.g. a free cron service like
cron-job.org) to keep it warm.
