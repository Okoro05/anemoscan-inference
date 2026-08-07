# AnemoScan Inference Service

A small standalone API that runs the trained AnemoScan model and returns a
predicted hemoglobin value. Deployed on Render (not Netlify) because it
needs a real persistent server, not a size-capped serverless function.

## Deploying to Render

1. Push this folder to its own GitHub repo (e.g. `anemoscan-inference`).
2. Put your trained `model.onnx` file inside the `model/` folder before pushing
   (or upload it directly on GitHub afterward, same as before).
3. Go to render.com, sign up (free), click **New +** -> **Web Service**.
4. Connect this GitHub repo.
5. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
6. Click **Create Web Service**. First deploy takes a few minutes.
7. Once live, Render gives you a URL like `https://anemoscan-inference.onrender.com`.
8. Test it: visit `https://your-url.onrender.com/health` in a browser — should show `{"status":"ok","model_present":true}`.

Note: Render's free tier spins the service down after inactivity, so the
first request after a while takes ~30-50 seconds to "wake up." Later
requests are fast.
