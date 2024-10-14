This repository contains a script that uses the Latent Diffusion Model (LDM) Super-Resolution pipeline to enhance low-resolution images. The script loads paired high-resolution and low-resolution images, fine-tunes the LDM pipeline, and visualizes the results. This project was executed on a Kaggle notebook environment with GPU support.

Table of Contents
Requirements
Dataset Preparation
Usage
Details
Custom Dataset
Super-Resolution Pipeline
Fine-Tuning and Visualization
Acknowledgements
Requirements
The following packages are required to run the script:

Python 3.7+
PyTorch
Torchvision
diffusers
PIL (Pillow)
Matplotlib
Kaggle environment with GPU support
Use the following command to install dependencies:

pip install torch torchvision diffusers pillow matplotlib
Dataset Preparation
Place your high-resolution images and low-resolution images in separate folders named:

/kaggle/working/high_res/high_res
/kaggle/working/low_res/final_low
Ensure that:

The high-resolution and low-resolution images have matching filenames.
The image pairs are properly prepared so they can be easily paired using the provided dataset class.
Usage
Clone the repository or copy the script to a Kaggle notebook environment.

Run the script to perform super-resolution and visualize results.

The script will process low-resolution images in small batches and generate their corresponding super-resolved outputs using the Latent Diffusion Model. The script will visualize some of the images after processing.

To run the script, you can execute it in any Python environment, though it's recommended to use Kaggle due to the GPU requirements.

Details
Custom Dataset
The ImagePairDataset class loads paired low-resolution and high-resolution images:

class ImagePairDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, transform=None):
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.transform = transform
        self.image_pairs = [(os.path.join(low_res_dir, f), os.path.join(high_res_dir, f))
                            for f in os.listdir(low_res_dir) if f in os.listdir(high_res_dir)]

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        low_res_path, high_res_path = self.image_pairs[idx]
        low_res_image = Image.open(low_res_path).convert("RGB")

        if self.transform:
            low_res_image = self.transform(low_res_image)

        return low_res_image, low_res_path
high_res_dir and low_res_dir specify the directories for high-resolution and low-resolution images.
The images are loaded in pairs and optionally transformed to tensors.
Super-Resolution Pipeline
The LDMSuperResolutionPipeline is used to enhance the low-resolution images:

model_id = "CompVis/ldm-super-resolution-4x-openimages"
ldm_pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
ldm_pipeline = ldm_pipeline.to(device)
The model CompVis/ldm-super-resolution-4x-openimages is used for the 4x super-resolution.
Fine-Tuning and Visualization
The function fine_tune_step runs inference on each image:

@torch.no_grad()
def fine_tune_step(image):
    return ldm_pipeline(image, num_inference_steps=50, eta=1).images[0]
The function takes a low-resolution image and performs super-resolution using the pipeline.
num_inference_steps controls the quality and speed of the output.
The process_batch function processes images in small batches to avoid memory issues:

def process_batch(batch_size=4):
    res_originals = []
    res_outputs = []
    
    for i, (low_res, low_res_path) in enumerate(dataloader):
        low_res = low_res.to(device)
        
        # Process image
        res_image = fine_tune_step(low_res)
        
        # Store results
        res_originals.append(low_res.cpu().squeeze(0).permute(1, 2, 0))
        res_outputs.append(res_image)
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        if (i + 1) % batch_size == 0:
            yield res_originals, res_outputs
            res_originals = []
            res_outputs = []
            gc.collect()  # Force garbage collection
Batches are processed to reduce memory footprint.
Images are visualized using matplotlib:
def plot_results(originals, outputs, num_samples=3):
    plt.figure(figsize=(15, 10))
    for i in range(min(num_samples, len(originals))):
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(originals[i])
        plt.title("Low-Resolution")
        plt.axis("off")

        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(outputs[i])
        plt.title("Super-Resolved Output")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
Acknowledgements
The LDM model used in this script is by CompVis and is hosted on the Hugging Face model hub: CompVis/ldm-super-resolution-4x-openimages.
The script is based on PyTorch and uses the diffusers library for efficient model handling.