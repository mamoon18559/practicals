#!git clone https://github.com/hardik0/AI-for-Medicine-Specialization
#%cd AI-for-Medicine-Specialization/AI-for-Medical-Diagnosis/

# 📦 Install required visualization and widget libraries
#!pip install itk itkwidgets ipywidgets

# 📦 Install required visualization and widget libraries

# 📚 Import necessary Python libraries
import numpy as np
import nibabel as nib  # for working with medical imaging data (.nii.gz)
import itk
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
import seaborn as sns

# 🎨 Set up default visualization styles

sns.set_style('darkgrid')

# ------------------------------
# 🧠 Load the MRI Brain Image
# ------------------------------

# Path to a sample 3D MRI image file
image_path = "BRATS_001.nii"

# Load the image using nibabel
image_obj = nib.load(image_path)
print(f"Loaded image object of type: {type(image_obj)}")

# Extract image data as a NumPy array
image_data = image_obj.get_fdata()
print(f"Image data type: {type(image_data)}")

# Check the shape of the image
height, width, depth, channels = image_data.shape
print(f"Image dimensions - Height: {height}, Width: {width}, Depth: {depth}, Channels: {channels}")

# ------------------------------
# 🖼️ Visualize a Random Slice
# ------------------------------

# Pick a random slice along the depth axis
i = np.random.randint(0, depth)
channel = 0  # Choosing the first channel

print(f"Showing slice {i} from channel {channel}")
plt.imshow(image_data[:, :, i, channel], cmap='gray')
plt.axis('off')
plt.show()

# ------------------------------
# 🔄 Interactive Slice Viewer
# ------------------------------

# Explore brain layers using a slider
def explore_3dimage(layer):
    plt.figure(figsize=(10, 5))
    plt.imshow(image_data[:, :, layer, 2], cmap='gray')
    plt.title(f'Layer {layer} - Channel 2', fontsize=16)
    plt.axis('off')
    plt.show()

# Create an interactive widget
interact(explore_3dimage, layer=(0, image_data.shape[2] - 1));


label_path = "BRATS_0011.nii"
label_obj = nib.load(label_path)

label_data = label_obj.get_fdata()

lh , lw , ld = label_data.shape
print(np.unique(label_data))

print("""
Label meanings:
0 = Normal tissue
1 = Edema
2 = Non-enhancing tumor
3 = Enhancing tumor
""")

layer = 55

classes_dict = {
    'Normal': 0.,
    'Edema': 1.,
    'Non-enhancing tumor': 2.,
    'Enhancing tumor': 3.
}

fig, ax = plt.subplots(nrows=1,ncols=4,figsize=(20,6))
for i,(label_name,label_val) in enumerate(classes_dict.items()):
    mask = np.where(label_data[:,:,layer]==label_val,255,0)
    ax[i].imshow(mask,cmap="gray")
    ax[i].axis('off')
plt.tight_layout()
plt.show()

select_class = ToggleButtons(
    options=list(classes_dict.keys()),
    description='Class:',
    button_style='info'
)
select_layer = IntSlider(min=0, max=label_data.shape[2]-1, description='Layer', continuous_update=False)

def plot_image(seg_class, layer):
    print(f"Layer {layer} - Label: {seg_class}")
    label_val = classes_dict[seg_class]
    mask = np.where(label_data[:, :, layer] == label_val, 255, 0)
    plt.figure(figsize=(8, 5))
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

interactive(plot_image,seg_class=select_class,layer=select_layer)
