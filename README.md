#Bio-Medical Image segmentation and Detection for Brain Tumour and Skin Lesion Diseases Through U-Net
#Tech Stack

 Frontend
 HTML
 CSS
 JavaScript

 Backend
 dijango Framework
 python

 Algorithm
 U-Net Deep Learning Algorithm


#Features

1. Disease Detection: Users can determine whether diseases are present in the uploaded image.

2. Location Identification: Users can identify the precise location of the disease within the uploaded image.

3. Size and Shape Recognition: Users can ascertain the size and shape characteristics of the disease present in the uploaded image.

4. Image Types: Users have the option to upload MRI images for detecting brain tumors and normal images for identifying skin lesions.

5. Region Detection: Users can visualize and comprehend the detected region of the disease within the input image.


#Steps To Run The Application

1. Open XAMPP and start both Apache and MySQL services.
2. Navigate to the directory "C:\WORKSPACE\Brain_Skin_Mask" using Command Prompt.
3. Run the following command to start the application:
"python manage.py runserver"
4. Copy the URL displayed in the Command Prompt after running the server.
5. Paste the copied URL into your web browser's address bar.
6. Once the application loads, test it by providing input images of MRI scans for brain tumor detection and skin images for skin lesion disease detection.


#Datasets Used in the Project

Our application focuses on predicting brain tumour locations using publicly accessible MRI datasets from The Cancer Imaging Archive (TCIA). 
These datasets contain manually annotated FLAIR abnormality segmentation masks from 110 patients in The Cancer Genome Atlas (TCGA) lower-grade glioma collection.
Each image, sized at 256x256 pixels, merges MRI slices from three modalities into RGB format. Furthermore, we leverage an additional dataset for skin lesion classification,
combining data from HAM10000 (2019) and MSLDv2.0, encompassing 14 different types of skin lesions.

Link For Brain Tumour Dataset :
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation

Link For Skin Lesion Dataset :
kaggle datasets download -d ahmedxc4/skin-ds



