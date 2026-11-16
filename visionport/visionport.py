import io
import cv2
import numpy as np
 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
 
# 1. Create the FastAPI application
app = FastAPI(title="Indian Traffic Detection API")
 
# 2. Allow the local Live Server webpage to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null"  # For a local file opened directly in a browser (demo only)
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)
 
# 3. Load the pretrained YOLOv8 Nano model
print("Loading YOLOv8 Nano model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully!")
 
# 4. Home route – used to check whether the API is running
@app.get("/")
async def home():
    return {
        "status": "Active",
        "message": "Indian Traffic Detection API is running!"
    }

# 5. Health-check route
@app.get("/health")
async def health():
    return {
        "status": "Active",
        "model": "YOLOv8 Nano",
        "device": "CPU"
    }
 
# 6. Image detection route
@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
 
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )
 
        # Convert image bytes into an OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
 
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )
 
        # Run YOLO inference on the CPU
        results = model.predict(
            source=img,
            imgsz=416,
            conf=0.25,
            device="cpu",
            verbose=False
        )
 
        # Draw boxes and class labels on the image
        annotated_img = results[0].plot()
 
        # Encode the annotated image as JPEG
        success, encoded_img = cv2.imencode(".jpg", annotated_img)
 
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Image encoding failed."
            )
 
        # Return the JPEG image to the browser
        return StreamingResponse(
            io.BytesIO(encoded_img.tobytes()),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"}
        )
 
    except HTTPException:
        raise
    except Exception as error:
        print("Detection error:", error)
        raise HTTPException(status_code=500, detail=str(error))
 
# 7. Optional direct-run entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
