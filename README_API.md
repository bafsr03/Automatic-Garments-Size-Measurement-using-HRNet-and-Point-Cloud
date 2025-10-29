## Garment Measurement API (MVP)

Features: JWT auth, rate-limiting, in-memory caching, structured JSON logging, API versioning (/v1), graceful errors, webhook callbacks, Docker setup.

### Run locally (Docker)

1) Create `.env` (optional):

```
JWT_SECRET=change-me
LOG_LEVEL=INFO
RATE_LIMIT_PER_MIN=60
RATE_LIMIT_BURST=30
CACHE_TTL_SECONDS=600
API_PREFIX=/v1
```

2) Build and run:

```
docker build -t garments-api -f api_app/../Dockerfile .
docker run -p 8000:8000 --env-file .env garments-api
```

### Usage

1) Health:
```
curl http://localhost:8000/v1/health
```

2) Get token:
```
curl -X POST http://localhost:8000/v1/auth/token
```

3) Process image (requires Authorization header):
```
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/token | python -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -H "Authorization: Bearer $TOKEN" \
  -F "image=@data/deepfashion2/validation/image/vest.jpg" \
  -F "category_id=5" \
  -F "true_size=L" \
  -F "true_waist=80" \
  -F "unit=cm" \
  http://localhost:8000/v1/process
```

Response:
```
{
  "measurement_vis": ".../api_runs/<id>/measure_vis_keypoints/000001.jpg",
  "size_scale": ".../api_runs/<id>/size_scale.json"
}
```

4) Webhook test receiver:
```
curl -X POST http://localhost:8000/v1/test/webhook-receiver -d '{"ping":1}' -H 'Content-Type: application/json'
```

Add `webhook_url` form field to /process to receive async callback with the same JSON.

