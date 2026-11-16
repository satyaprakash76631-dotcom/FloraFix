# ANISH17LIVE LONG — Plant Disease Detector

A beginner-friendly Flask website with:
- Live camera access (where supported by the browser)
- Take-a-photo button
- Upload from device
- AI image classification endpoint
- Possible effects, general care steps, and prevention advice

## Important: the model is a separate training step

The website code does not contain a trained plant-disease model. You must train one using a correctly labelled dataset before predictions will work. The starter training script expects a PlantVillage-style dataset with one folder per class. A model trained on lab-style leaf images may be less accurate on real garden photos, so treat results as a screening suggestion, not a guaranteed diagnosis.

## 1. Install Python

Install Python 3.10 or 3.11. During installation on Windows, tick **Add Python to PATH**.

## 2. Open the project

Extract the ZIP. Open the `anish17live_long` folder in VS Code or PyCharm.

Open the terminal in that folder and run:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

TensorFlow installation can take time and needs a compatible Python version. If installation fails, use Python 3.10/3.11 in a fresh virtual environment.

## 3. Add a labelled plant-leaf dataset

Download a permitted PlantVillage dataset from a trusted source such as Kaggle. Check the dataset's licence and terms before using it publicly.

Create this structure:

```text
anish17live_long/
  dataset/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___healthy/
    Tomato___Early_blight/
    Tomato___healthy/
    ...
```

Put each image inside its correct disease/healthy class folder. Do not mix classes. The folder names become the model's labels.

## 4. Train the AI model

Run:

```bash
python train_model.py
```

When training finishes, these two files should appear beside `app.py`:

- `plant_disease_model.keras`
- `class_names.json`

Training can take a while. A GPU or Google Colab may help with larger datasets.

## 5. Start the website

Run:

```bash
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000
```

Click **Open live camera**, allow permission, point the camera at a leaf, click **Take photo**, then **Analyse leaf**. Or choose a photo from your device.

Camera access normally requires `localhost` or HTTPS. If camera access does not work, use the photo upload button.

## 6. Put it online

For a public website, deploy the Flask app to a Python hosting service that supports your TensorFlow model and enough memory. Add the trained model and class_names.json as deployment files, configure the start command as:

```bash
gunicorn app:app
```

Add `gunicorn` to `requirements.txt` before deploying. Use HTTPS so visitors can access their camera. Never upload private photos or personal data without explaining how they are handled.

## Project files

- `app.py` — Flask server and prediction API
- `train_model.py` — trains a MobileNetV2 image classifier
- `templates/index.html` — website page
- `static/style.css` — design
- `static/app.js` — camera, capture, upload, and submit logic
- `requirements.txt` — Python dependencies

## Accuracy and plant-care safety

A model's confidence score is not proof that the diagnosis is correct. Similar spots can have different causes (fungus, bacteria, pests, nutrient issues, watering stress, or weather). The included care tips are general. Confirm the plant species and diagnosis with a local agriculture/horticulture expert before applying chemicals or removing large parts of a plant.


## 7. Deployment preparation (Render)

This project includes `render.yaml` and Gunicorn for a Render web-service deployment.
Before deployment:
1. Train the model locally or in a suitable notebook environment.
2. Ensure `plant_disease_model.keras` and `class_names.json` are included in the deployment repository.
3. Push the project to a GitHub repository.
4. In Render, create a new Web Service and connect that repository. Render can detect `render.yaml`.
5. Confirm the build command is `pip install -r requirements.txt` and start command is `gunicorn app:app`.
6. After the service is live, open its `onrender.com` URL and test it in a browser.

TensorFlow can require substantial memory and may exceed the limits of small/free hosting plans. If the build or service runs out of memory, use a hosting plan with suitable RAM or host inference separately. Do not publish until the trained model has been tested with real leaf photos.

The app provides `/robots.txt` and `/sitemap.xml` for search-engine discovery. After deployment, add the exact live URL to Google Search Console and Bing Webmaster Tools, verify ownership, and submit `https://YOUR-LIVE-DOMAIN/sitemap.xml`. Search engines decide when and whether to index a site; submission is not an instant listing guarantee.
