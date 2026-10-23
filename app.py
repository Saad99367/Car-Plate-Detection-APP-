import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load your trained car plate detection model
# Replace 'best.pt' with the path to your actual model weights
model = YOLO("models/CarPlateYOLO.pt")

def resize_for_display(crop, target_height=120):
    """
    Resizes the cropped plate to a standard readable height while 
    preserving the aspect ratio so the plate never looks stretched.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return crop
    
    scale = target_height / float(h)
    new_w = int(w * scale)
    
    # Use INTER_CUBIC for upscaling small crops (sharper text), INTER_AREA for downscaling
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(crop, (new_w, target_height), interpolation=interpolation)

def detect_and_crop_plate(image, conf_threshold, padding_pct):
    if image is None:
        return None, None, []

    # Make a copy for drawing bounding boxes (Gradio passes RGB images)
    annotated_img = image.copy()
    img_h, img_w = image.shape[:2]

    # 2. Run inference
    results = model.predict(image, conf=conf_threshold)[0]
    
    cropped_plates = []

    for box in results.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # 3. Draw bounding box and confidence score on the main image
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Plate {conf:.2f}"
        cv2.putText(
            annotated_img, label, (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        # 4. Calculate padding so the crop doesn't cut off plate edges
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * (padding_pct / 100.0))
        pad_y = int(box_h * (padding_pct / 100.0))

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(img_w, x2 + pad_x)
        crop_y2 = min(img_h, y2 + pad_y)

        # 5. Crop and resize to a clean, uniform display size
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        good_size_crop = resize_for_display(crop, target_height=140)
        
        cropped_plates.append((good_size_crop, f"Confidence: {conf:.2f}"))

    # Return the primary crop for the single-view box, plus all crops for the gallery
    best_crop = cropped_plates[0][0] if cropped_plates else None

    return annotated_img, best_crop, cropped_plates

# 6. Build the Gradio UI
with gr.Blocks(title="Car Plate Detector") as demo:
    gr.Markdown("## 🚗 Car License Plate Detection & Cropping")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Car Image")
            
            with gr.Accordion("Detection & Crop Settings", open=True):
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.4, step=0.05,
                    label="Confidence Threshold"
                )
                pad_slider = gr.Slider(
                    minimum=0, maximum=30, value=8, step=1,
                    label="Crop Padding (%)",
                    info="Adds extra margin around the plate so letters aren't cut off"
                )
                
            submit_btn = gr.Button("Detect & Crop Plate", variant="primary")

        with gr.Column(scale=1):
            output_annotated = gr.Image(type="numpy", label="Detected Image with Bounding Box")
            
            # Dedicated box for the main cropped plate with a fixed, clean container height
            output_best_crop = gr.Image(
                type="numpy", 
                label="Cropped License Plate (Normalized Size)", 
                height=180
            )
            
            # Gallery in case an image has multiple cars/plates
            output_gallery = gr.Gallery(
                label="All Detected Plates", 
                columns=2, 
                height="auto"
            )

    submit_btn.click(
        fn=detect_and_crop_plate,
        inputs=[input_image, conf_slider, pad_slider],
        outputs=[output_annotated, output_best_crop, output_gallery]
    )

if __name__ == "__main__":
    demo.launch()