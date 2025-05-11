# Indian Comic Style Transfer

> Fine‑tuning Stable Diffusion to generate images in the bold, vibrant style of 1980s Raj Comics’ _Nagraj_.

## 📖 Overview

This project explores how to adapt a pre‑trained Stable Diffusion model to the unique visual grammar of _Indian_ comic panels. All details—goals, approach, and evaluation metrics—are experimental and may evolve as we gain insights.

## 🎯 Project Goals (Tentative)

-   Experiment with LoRA adapters to capture Nagraj’s high‑contrast line art and flat colors.
-   Generate photo‑realistic inputs rendered as dynamic comic panels in Nagraj style.
-   Build an end‑to‑end pipeline for easy inference and sharing.
-   Evaluate style fidelity qualitatively; refine based on visual assessments.

## 💡 Project Idea

Leverage unpaired style transfer: fine‑tune Stable Diffusion exclusively on cropped panels from the Raj Comics _Nagraj_ series. By anchoring with a common style prompt, we teach the model to apply bold outlines, vibrant fills, and dramatic compositions characteristic of these comics.

## 🛠 Approach (Initial Plan)

1. **Panel Extraction**: Use a Python/OpenCV script to crop rectangular panels from the PDF.
2. **Data Preparation**: Resize and center‑crop all panels to 512×512 px; store in `data/style/`.
3. **Prompt Assignment**: Use 一 or a small set of style‑focused prompts to label each image.
4. **LoRA Fine‑tuning**: Run the Hugging Face Diffusers LoRA example on these panels.
5. **Inference & Review**: Generate test outputs in Colab; compare against original panels.
6. **Iteration**: Tweak prompts, hyperparameters, and dataset composition based on results.

## 🚧 Current State

-   **Panel extraction script**

    -   Renders the PDF to images
    -   Crop comic panels

-   **LoRA training recipe**
    -   Based on Hugging Face Diffusers

## 🖼️ Image Processing

This project includes tools for processing comic images, such as panel extraction and analysis, using different approaches.

### Deep Learning Approach

-   **Functionality**: Utilizes a deep learning model for tasks like comic panel segmentation.
-   **Inference Script**: The core logic can be found in `image_processing/using_deep_learning/inference.py`.
-   **Sample Images**: Example images for this approach are in `image_processing/using_deep_learning/samples/`.
-   **Source & Methodology**: Based on the [Vrroom/segment-anything-comic](https://github.com/Vrroom/segment-anything-comic) repository, which adapts the Segment Anything Model (SAM).

### OpenCV Approach

-   **Functionality**: Provides utilities for image operations, primarily focused on panel extraction from comic pages.
-   **Script**: Panel extraction logic is available in `image_processing/using_opencv/extract_panels.py`.
-   **Sample Images**: Related sample images can be found in `image_processing/using_opencv/samples/`.

### Advanced Comic Analysis with Magi Model

For more comprehensive understanding and transcription of comic/manga pages, the **Magi (The Manga Whisperer)** model presents a specialized and powerful solution.

-   **Capabilities**: Magi is designed to detect not only panels but also characters and text blocks. It further processes this information by ordering panels, performing OCR, clustering character appearances, and matching dialogue to speakers, enabling the generation of detailed transcripts.
-   **Advantages over other approaches**:
    -   **Specialization**: Unlike generic segmentation models or rule-based OpenCV methods, Magi is tailored for the specific structures and content of comics and manga.
    -   **Rich Information Extraction**: It goes beyond simple object detection to provide a deeper understanding of the page, including character interactions and dialogue flow.
    -   **End-to-End Processing**: Offers a more complete pipeline from image to structured textual output.
-   **Resources**:
    -   **GitHub Repository**: [ragavsachdeva/magi](https://github.com/ragavsachdeva/magi)
    -   **Hugging Face Model**: [ragavsachdeva/magi](https://huggingface.co/ragavsachdeva/magi) (and its v2 version)

This model could be particularly beneficial for tasks requiring automated content analysis, character tracking, or generating accessible versions of comic content.

## Model comming soon
