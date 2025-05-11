# Indian Comic Style Transfer

> Fine‑tuning Stable Diffusion to generate images in the bold, vibrant style of 1980s Raj Comics’ *Nagraj*.

## 📖 Overview

This project explores how to adapt a pre‑trained Stable Diffusion model to the unique visual grammar of *Nagraj* comic panels. All details—goals, approach, and evaluation metrics—are experimental and may evolve as we gain insights.

## 🎯 Project Goals (Tentative)

* Experiment with LoRA adapters to capture Nagraj’s high‑contrast line art and flat colors.
* Generate photo‑realistic inputs rendered as dynamic comic panels in Nagraj style.
* Build an end‑to‑end Colab/demo pipeline for easy inference and sharing.
* Evaluate style fidelity qualitatively; refine based on visual assessments.

## 💡 Project Idea

Leverage unpaired style transfer: fine‑tune Stable Diffusion exclusively on cropped panels from the Raj Comics *Nagraj* series. By anchoring with a common style prompt, we teach the model to apply bold outlines, vibrant fills, and dramatic compositions characteristic of these comics.

## 🛠 Approach (Initial Plan)

1. **Panel Extraction**: Use a Python/OpenCV script to crop rectangular panels from the PDF.
2. **Data Preparation**: Resize and center‑crop all panels to 512×512 px; store in `data/style/`.
3. **Prompt Assignment**: Use one or a small set of style‑focused prompts to label each image.
4. **LoRA Fine‑tuning**: Run the Hugging Face Diffusers LoRA example on these panels.
5. **Inference & Review**: Generate test outputs in Colab; compare against original panels.
6. **Iteration**: Tweak prompts, hyperparameters, and dataset composition based on results.

> **Note**: All steps are subject to change as we learn what works best for this niche comic style.

## 🚧 Current State

* **Panel extraction script**

  * Renders the PDF to images
  * Crop comic panels
* **LoRA training recipe** 
  * Based on Hugging Face Diffusers


## Model comming soon
