from transformers import YolosForObjectDetection, YolosImageProcessor
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
# Load the model and processor
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

# Load and process the image


def detect_object(image):
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
    return final

images_list=[]
foldername = 'Ajude images'
for filename in os.listdir(foldername):
    if filename.endswith(('.png','.jpg','.jpeg')):
        image = Image.open(os.path.join(foldername,filename))
        images_list.append(image)
# print(images_list)

# Assuming image_list is already defined and contains your images
for i in range(2):
    plt.imshow(images_list[i])
    plt.title(f'Image {i + 1}')
    plt.axis('off')  # Hide the axis for a cleaner look
    plt.show()  # Display the image
    print(detect_object(images_list[i]))  # Assuming this function returns some detection results
