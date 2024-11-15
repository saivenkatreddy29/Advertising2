from transformers import YolosForObjectDetection, YolosImageProcessor
import torch
from PIL import Image

# Load the model and processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

# Load and process the image
image = Image.open("Basic Test images/000079.png")
inputs = image_processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the results
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

# Get the top k detections
k = 5  # Adjust this value to get more or fewer top detections
top_k_indices = results["scores"].argsort(descending=True)[:k]
final = set()

for i in top_k_indices:
    box = results["boxes"][i].tolist()
    score = results["scores"][i].item()
    label = model.config.id2label[results["labels"][i].item()]
    # print(f"Detected {label} with confidence {score:.2f} at location {box}")
    if score>0.50:
        final.add(label)
print(final)