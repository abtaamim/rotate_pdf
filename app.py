from typing import List
import torchvision
import PIL

def pred_and_plot_image(model:torch.nn.Module ,
                       image:PIL.Image.Image,
                       class_names: List[str]= None,
                       transform=None):
    #target_image= TF.to_grayscale(target_image, num_output_channels=1),
    #target_image= torchvision.io.read_image(str(image_path)).type(torch.float32)
    image = image.convert("L")
    target_image= TF.to_tensor(image).type(torch.float32)
    #target_image= target_image/255
    if transform:
        target_image= transform(target_image)
        temp= target_image
    model.eval()
    with torch.inference_mode():
        target_image= target_image.unsqueeze(0)
        target_image_pred= model(target_image)

    target_image_pred_probs= torch.softmax(target_image_pred, dim=1)
    target_image_pred_label= torch.argmax(target_image_pred_probs, dim=1)
    #plt.imshow(TF.to_pil_image(temp))
    if class_names:
        title= class_names[target_image_pred_label]
        #f"Pred: {class_names[target_image_pred_label]} | prob: {target_image_pred_probs.max()}"
    else:
        title= target_image_pred_label
        #f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max()}"
    # plt.title(title)
    # plt.axis(False)
    return title #title is the prediction


from PIL import Image
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import numpy as np
import io
def cropp_images_from_pdf (pdf_file):
    reader = easyocr.Reader(['en','bn'])
    # pdf_path= "/kaggle/input/test-pdf-2/2025-05-17 01_13_43-DSD note (Arafat).pdf - Adobe Acrobat Reader (64-bit).pdf"
    # doc = fitz.open(pdf_path)
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_images = [] 
    
    all_pages_data = [] # save page numbers and coordinate of words
    real_page_images=[]
    for page_number, page in enumerate(doc, start=1):
        
        print(f"working on page {page_number}")
        pix = page.get_pixmap(dpi=400)
        img = Image.frombytes("RGB", [width, height], pix.samples)
    
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, dpi=(400, 400))
        buf.seek(0)  # Rewind buffer
        real_page_images.append(buf)
        
        img = img.convert("L")
        page_images.append(img)
        
        img_np = np.array(img)
        
        #reading one page
        results= reader.readtext(img_np) 
        
        page_info={
            "page": page_number,
            "words":[]
        }
        for bbox, text, confidence in results:
            word_data = {
                #"text": text,
                #"confidence": confidence,
                "box": bbox  # list of 4 points
            }
            page_info["words"].append(word_data)
            
        all_pages_data.append(page_info)
    all_pages_with_cropped_word=[]
    
    for i, page_info in enumerate(all_pages_data):
        crop_info={
            "page": page_info["page"],
            "cropped_images": []
        }
        print(f"cropping from page {crop_info['page']} ")
        
        for word_data in page_info["words"]:
            box = word_data["box"]
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            cropped_image = page_images[i].crop((x_min, y_min, x_max, y_max))
            crop_info["cropped_images"].append(cropped_image)
    
        all_pages_with_cropped_word.append(crop_info)

    return all_pages_with_cropped_word, real_page_images

def rotation_prediction_of_cropped_images(all_pages_with_cropped_word):
    cropped_rotation_info=[]
    class_names=[0,1,2,3]
    custom_image_transform= transforms.Compose([
        transforms.Resize(size=(32, 32))
    ])
    for i, crop_info in enumerate(all_pages_with_cropped_word):
        cropped_images= crop_info["cropped_images"]
        for cropped_image in cropped_images:
            
            rotation=pred_and_plot_image(model= new_model,
                       image=cropped_image,
                       class_names= class_names,
                       transform=custom_image_transform,
                       )
            rotation_info={
                "page":crop_info["page"],
                "rotation":rotation
            }
            cropped_rotation_info.append(rotation_info)

    return cropped_rotation_info
#         plt.imshow(cropped_image)
#         plt.title(rotation)
#         plt.axis('off')
#         plt.show()
# for i, rotation_info in enumerate(cropped_rotation_info):
#     print(f"page: {rotation_info['page']} | rotation: {rotation_info['rotation'] }")

from collections import Counter
def group_rotations_by_page(cropped_rotation_info):
    # Create a dictionary to store rotations grouped by page
    rotations_by_page = {}
    
    for info in cropped_rotation_info:
        page = info["page"]
        rotation = info["rotation"]
        
        # If page is not in dictionary, add it with an empty list
        if page not in rotations_by_page:
            rotations_by_page[page] = []
        
        # Append the rotation to the corresponding page
        rotations_by_page[page].append(rotation)
    
    return rotations_by_page


def get_most_common_rotation_by_page(cropped_rotation_info):

    rotations_by_page = group_rotations_by_page(cropped_rotation_info)
    
    # Create a result dictionary to store the most common rotation for each page
    most_common_rotations = []
    
    # Find the most common rotation for each page
    for page, rotations in rotations_by_page.items():
        counter = Counter(rotations)
        most_common = counter.most_common(1)[0][0]  # Get the most common rotation value
        most_common_rotations.append(most_common)
    
    return most_common_rotations
    #, rotations_by_page
    
most_common_rotations= get_most_common_rotation_by_page(cropped_rotation_info)

# for page in range( len(most_common_rotations)):
#     print(f" page: {page} | common rotation: {most_common_rotations[page]}")

import cv2
import numpy as np
from IPython.display import FileLink
def final_rotated_pdf(most_common_rotations):
    for page in range(len(most_common_rotations)):
        print(f"page {page}")
        img = real_page_images[page]  # This is either a BytesIO (initially) or a PIL.Image (after first rotation)
    
        # Only open if it's a BytesIO
        if isinstance(img, io.BytesIO):
            img = Image.open(img)
        
        img_np = np.array(img)
        k = most_common_rotations[page]  # 0, 1, 2, or 3
        rotated_np = np.rot90(img_np, k=k)
        rotated_img = Image.fromarray(rotated_np)
        print(f"Size after saving: width = {rotated_img.width}, height = {rotated_img.height}")
        if most_common_rotations[page] == 1 or most_common_rotations[page] == 3 :
            rotated_img= rotated_img.resize((rotated_img.height, rotated_img.width))
        real_page_images[page] = rotated_img
        
      
    pdf_images = []
    for img in real_page_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pdf_images.append(img)
    
    # Save all pages into a single PDF
    output_pdf_path = "rotated_output.pdf"
    pdf_images[0].save(
        output_pdf_path,
        save_all=True,
        append_images=pdf_images[1:]
    )

    return output_pdf_path
    # FileLink("rotated_output.pdf")

# for img in real_page_images:
#     plt.imshow(img)
#     plt.show()
# len(real_page_images)

def prediction_and_final_pdf(pdf_file ):
    all_pages_with_cropped_word, real_page_images = cropp_images_from_pdf(pdf_file)
    cropped_rotation_info = rotation_prediction_of_cropped_images(all_pages_with_cropped_word)
    most_common_rotations = group_rotations_by_page(cropped_rotation_info)
    output_pdf_path = final_rotated_pdf(most_common_rotations)
    return output_pdf_path