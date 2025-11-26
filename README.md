# Underwater Images Enhancement

Enhancing underwater images to restore natural color, improve visibility, and recover details often lost due to light absorption and scattering in underwater environments.

---

## 🚀 Project Overview

Underwater images often suffer from:

* Color distortion (especially loss of red channel)
* Low contrast
* Noise and hazy appearance
* Detail degradation

This project provides a lightweight enhancement workflow to improve the visual quality of underwater images.

---

## ✨ Features

* Simple and effective underwater image enhancement
* Color correction and contrast improvement
* Batch processing support (folder-based)
* Example input image included

---

## 📂 Project Structure

```
Underwater-Images-Enhancement/
├── unet/              # Main enhancement algorithm (or model, optional)
├── sample.png         # Sample input image
└── README.md          # Documentation
```

---

## 🖼 Example

### Input Image

![Sample Input](sample.png)

> Enhanced results will be added soon.

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/pinwei123/Underwater-Images-Enhancement.git
cd Underwater-Images-Enhancement
```

### 2️⃣ Install Dependencies

(Modify based on your actual package usage)

```bash
pip install numpy opencv-python matplotlib
```

---

## ▶️ Usage

Example command (modify based on your script structure):

```bash
python unet/main.py \
    --input_folder path/to/your/underwater_images \
    --output_folder path/to/save/enhanced_images
```

---

## 🛠 Tech Stack

* Python 3.x
* OpenCV for image processing
* NumPy for array operations
* (Optional) PyTorch / TensorFlow if deep learning enhancement is added later

---

## 📌 Roadmap

Planned improvements:

* Support configurable enhancement parameters
* Add deep learning-based enhancement method (U-Net or Diffusion)
* Provide more benchmark examples (before/after)
* Performance evaluation metrics for underwater image quality

---

## 🤝 Contributing

Contributions are welcome!
Feel free to submit issues or pull requests.

---

## 📄 License

This project currently does not include a specific license.
(You may add MIT License or others based on future needs.)

---

## 🙌 Acknowledgements

Thanks to the underwater imaging research community for dataset inspiration and enhancement methodologies.

---
