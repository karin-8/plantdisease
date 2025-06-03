import os
from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import transforms
from PIL import Image
# import dino
def classify_image(image):
    # Define supported image extensions
    global IMAGE_EXTENSIONS
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg'}
    root_directory = "rice diseases/rice diseases"
    # Find all images
    images = find_images_in_directory(root_directory)
    # return images
    # Create the dataset and dataloader
    # image_paths = [os.path.join("data/images", img) for img in os.listdir("data/images")]
    image_paths = list(images)
    embeddings = np.load("embeddings.npy")
    global disease_list
    disease_list = sorted(list(set(["_".join(im.split("/")[2].split("_")[:-1]) for im in images])))


    

    
    y_train = [name_to_class(im) for im in images]
    X_train = embeddings
    
    # Prepare training data
    # X_train, y_train = zip(*labeled_training_data)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
            
    return disease_list[classify_or_flag_unknown(image, X_train, knn)]
    
def embed_image(image):
    # Load pretrained DINO model
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    with torch.no_grad():
        embedding = model(image).flatten().cpu()  # Flatten the output to get a 1D embedding
    return embedding
    
def name_to_class(name):
  return disease_list.index("_".join(name.split("/")[2].split("_")[:-1]))


def classify_or_flag_unknown(image, X_train, knn):
    new_embedding = embed_image(image)
    similarity_scores = cosine_similarity([new_embedding], X_train)
    max_similarity = np.max(similarity_scores)

    if max_similarity < 0.5:  # Example threshold
        return "Unknown"
    else:
        return knn.predict([new_embedding])[0]

    
def find_images_in_directory(root_dir):
    """
    Recursively find all image files in the specified directory and its subdirectories.
    :param root_dir: The root directory to search.
    :return: A list of image file paths.
    """
    image_files = []

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(str(file_path))

    return image_files
