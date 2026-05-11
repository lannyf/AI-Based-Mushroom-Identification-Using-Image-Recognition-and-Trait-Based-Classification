# Tutorial: How SAM 2 and YOLOv8 Work Together

This document explains the theory and general practice behind the two-stage segmentation pipeline used in this project. It is written for anyone who needs to understand why these models are used and how they interact especially useful for thesis writing.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [What Is Image Segmentation?](#2-what-is-image-segmentation)
3. [SAM 2: The Zero-Shot Segmenter](#3-sam-2-the-zero-shot-segmenter)
4. [YOLOv8: The Fast Detector](#4-yolov8-the-fast-detector)
5. [Why Use Both?](#5-why-use-both)
6. [The Full Pipeline Step-by-Step](#6-the-full-pipeline-step-by-step)
7. [Key Concepts Explained](#7-key-concepts-explained)
8. [Common Misconceptions](#8-common-misconceptions)

---

## 1. The Big Picture

You have two AI models that solve the same problem find the mushroom in a photo but they are optimized for opposite ends of a trade-off:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **SAM 2** | Very slow (3-8 sec) | Very high | Offline data preparation |
| **YOLOv8** | Very fast (~100 ms) | Good enough | Real-time app inference |

**The core idea:** Use the slow, accurate model (SAM 2) to create training data. Teach the fast model (YOLOv8) using that data. Deploy only the fast model.

Think of it like a master craftsman (SAM 2) teaching an apprentice (YOLOv8). The master is too slow to work on the assembly line, but they can prepare detailed instructions. The apprentice learns from those instructions and works fast enough for real production.

---

## 2. What Is Image Segmentation?

Before understanding the models, you need to understand the task.

### Classification vs. Detection vs. Segmentation

| Task | Output | Example |
|------|--------|---------|
| **Classification** | Label only | "This is a mushroom" |
| **Detection** | Bounding box | "Mushroom is at coordinates [x1, y1, x2, y2]" |
| **Segmentation** | Pixel mask | "Every pixel belonging to the mushroom is marked white" |

### Why Segmentation Matters for Your App

Your trait extractor needs to measure:
- **Cap color** - Is it red, brown, yellow?
- **Shape** - Is it convex, flat, funnel-shaped?
- **Texture** - Smooth, scaly, or wrinkled?

If you measure these across the **entire photo**, the green grass in the background contaminates the color analysis. The forest floor adds false texture. A hand holding the mushroom introduces skin-tone pixels.

**Segmentation isolates the mushroom pixels** so the trait extractor analyzes only what matters.

---

## 3. SAM 2: The Zero-Shot Segmenter

### What SAM 2 Is

SAM 2 (Segment Anything Model 2) is a foundation model developed by Meta. It can segment virtually any object in any image without ever being trained on that specific object class.

This property is called **zero-shot learning**: the model has never seen a mushroom during its training, yet it can segment mushrooms because it learned general "objectness" concepts from millions of diverse images.

### How SAM 2 Works (Simplified)

SAM 2 has two main components:

1. **Image Encoder** - A large neural network that processes the entire image and builds a rich understanding of every region. It asks: "What objects are here, and where are their boundaries?"

2. **Mask Decoder** - A lightweight network that generates the actual mask. You give it a **prompt** (a hint about what you want), and it returns the segmentation mask for that object.

### Prompts: How You Tell SAM 2 What to Segment

SAM 2 does not guess what you want. You must give it a prompt:

| Prompt Type | What You Provide | Best For |
|-------------|------------------|----------|
| **Point** | Click on the object | Quick, single objects |
| **Box** | Draw a rectangle around the object | Objects near edges or cluttered scenes |
| **Point + negative points** | Click object + click background to exclude | Reducing false inclusions |

In your project, the pipeline uses:
- A **center point** with **4 corner negative points** ("segment the thing in the middle, not the edges")
- A **bounding box fallback** from generic YOLO ("here is a rough area, segment inside it")

### Why SAM 2 Is Not Used at Runtime

SAM 2's image encoder is massive. On a CPU, it takes 3-8 seconds per image. A user taking a photo with your app will not wait 5 seconds for a mask to appear.

**SAM 2 is an offline tool.** It runs once, before deployment, to generate training masks.

---

## 4. YOLOv8: The Fast Detector

### What YOLOv8 Is

YOLO (You Only Look Once) is a family of real-time object detection and segmentation models. YOLOv8 is the 8th generation, developed by Ultralytics.

The name "You Only Look Once" describes how it works: the model processes the entire image in a single forward pass through the network. Older methods slide a window across the image thousands of times; YOLO looks once and predicts everything simultaneously.

### How YOLOv8 Segmentation Works

YOLOv8-seg adds a segmentation head to the standard detection network:

1. **Backbone** - Processes the image through convolutional layers to extract features
2. **Neck** - Combines features at different scales (small details + large context)
3. **Detection Head** - Predicts bounding boxes and class labels
4. **Segmentation Head** - Predicts a prototype mask and coefficients, then combines them into per-instance masks

The output is a bounding box, a class label (mushroom), and a binary mask all in one pass.

### Why YOLOv8 Needs Fine-Tuning

The default yolov8n-seg.pt model is pre-trained on the **COCO dataset**, which contains 80 common classes: people, cars, dogs, chairs, etc. **COCO does not contain mushrooms.**

When the generic model sees a mushroom photo, it has no concept of "mushroom." It might detect:
- A hand (because it knows "person")
- An orange blob (because it knows "orange")
- Nothing at all

**Fine-tuning** teaches the model a new class using your mushroom dataset. It keeps the general visual knowledge from COCO (edges, textures, shapes) and adds domain-specific knowledge (cap contours, stem silhouettes, mushroom color patterns).

### Why YOLOv8 Is Fast

- **Single forward pass:** No sliding windows, no region proposals
- **Tiny variant (n):** yolov8n-seg has only ~3.2 million parameters it fits in 6.8 MB
- **Optimized for edge devices:** Runs at 30-100 frames per second on GPU, ~100 ms on CPU

---

## 5. Why Use Both?

You might ask: "If SAM 2 is so good, why not just use it? If YOLOv8 is so fast, why not just use it?"

### SAM 2 Alone
- Near-perfect masks
- Zero-shot (no training data needed)
- Too slow for real-time use (3-8 seconds)
- Requires a prompt for every image

### YOLOv8 Alone
- Extremely fast (~100 ms)
- Fully automatic (no prompts)
- Small file size (6.8 MB)
- Needs training data to learn new classes
- Less accurate boundaries than SAM 2

### Together
| Stage | Model | Role | When It Runs |
|-------|-------|------|--------------|
| **Offline** | SAM 2 | Generates high-quality pseudo-label masks | Once, before training |
| **Training** | YOLOv8 | Learns from SAM 2's masks via fine-tuning | Once, in Google Colab |
| **Runtime** | YOLOv8 | Segments user photos in real-time | Every time a user takes a photo |

---

## 6. The Full Pipeline Step-by-Step

Here is the complete data flow, from raw images to deployed model:

### Step 1: Generate Pseudo-Labels with SAM 2

Input:  352 training images (data/raw/images/)
Tool:   SAM 2 (zero-shot, no training)
Output: 352 binary masks (data/SegMaskSAM2/)

For each image:
1. SAM 2 receives a prompt (center point + negative corners)
2. It proposes 3 candidate masks
3. A ranking heuristic selects the best one (prompt overlap, compactness, border touch)
4. The mask is saved as a PNG file

These masks are called **pseudo-labels** because they are machine-generated approximations of ground truth, not human-verified.

### Step 2: Prepare YOLO Training Dataset

Input:  352 SAM 2 masks + 48 background images
Tool:   prepare_yolo_seg_dataset.py
Output: data/segmentation/ (YOLO format)

The script:
1. Pairs each training image with its SAM 2 mask
2. Converts binary masks to YOLO polygon format (normalized coordinates)
3. Simplifies polygons using Ramer-Douglas-Peucker (reduces vertex count)
4. Performs an 80/20 train/validation split
5. Adds background images with empty labels (teaches "no mushroom here")

### Step 3: Fine-Tune YOLOv8 in Google Colab

Input:  data/segmentation/ (YOLO dataset)
Tool:   YOLOv8n-seg on T4 GPU
Output: artifacts/yolov8_seg_ft.pt (fine-tuned model)

Training details:
- Starts from COCO-pretrained weights
- Learns single class: 0 = mushroom
- 100 epochs with early stopping (often stops at ~60)
- Strong color augmentation to handle variable forest lighting

### Step 4: Evaluate Against Ground Truth

Input:  Fine-tuned YOLO + evaluation images + ground-truth masks
Tool:   evaluate_segmentation.py
Output: segmentation_evaluation.json (IoU, Precision, Recall)

The evaluation compares YOLO's predicted masks against held-out ground truth (either SAM 2 masks or manual annotations).

### Step 5: Deploy

Runtime: mushroom_segmenter.py loads artifacts/yolov8_seg_ft.pt
Input:   User photo
Output:  Binary mask isolating the mushroom

If the fine-tuned model is missing, it falls back to the generic yolov8n-seg.pt.

---

## 7. Key Concepts Explained

### Zero-Shot Learning
The ability to perform a task without any task-specific training data. SAM 2 was trained on millions of images with diverse objects; it learned generalizable concepts like "objects have boundaries" and "objects are contiguous regions." It can segment mushrooms without ever seeing one during training.

### Pseudo-Labels
Machine-generated labels used as if they were ground truth. SAM 2's masks are pseudo-labels. They are ~85-95% accurate good enough to train another model, but not perfect. Errors in pseudo-labels propagate to the student model (YOLOv8), which is why manual annotations improve results.

### Transfer Learning
Taking a model trained on one task (COCO object detection) and adapting it to a new task (mushroom segmentation). YOLOv8 keeps its learned visual features (edges, corners, textures) and only retrains the final layers to recognize mushrooms. This is why fine-tuning works with only ~350 images instead of millions.

### Intersection over Union (IoU)
The metric for segmentation quality. See the main guide for details. In short: IoU = 1.0 means perfect overlap between predicted mask and ground truth; IoU = 0.0 means no overlap.

### Background Images
Images that contain no mushroom at all. They are given empty label files. During training, YOLO learns that these images should produce no detections. This dramatically reduces false positives (hallucinated mushrooms in grass or soil).

---

## 8. Common Misconceptions

### "I trained SAM 2"
**Truth:** SAM 2 was pre-trained by Meta. You only ran it in zero-shot mode. You did not train it. Training SAM 2 from scratch requires millions of images and weeks of GPU time.

### "More YOLO epochs = better model"
**Truth:** YOLOv8 has early stopping. If validation metrics stop improving, additional epochs cause overfitting (the model memorizes training images instead of learning general patterns).

### "SAM 2 masks are perfect ground truth"
**Truth:** SAM 2 makes mistakes especially on hands, unusual angles, and cluttered backgrounds. It is excellent, but not infallible. That is why manual annotations improve the pipeline.

### "I should crop training images tightly around mushrooms"
**Truth:** Cropping removes context. The model needs to learn that mushrooms appear at different scales, partially occluded, and surrounded by various backgrounds. Keep full frames.

### "The CNN and YOLO compete with each other"
**Truth:** They solve completely different problems. The CNN classifies which species is in the photo. YOLO segments where the mushroom is in the photo. They complement each other.

---

## Summary

> **SAM 2** is the slow, smart teacher. It generates high-quality training masks offline.
>
> **YOLOv8** is the fast, lightweight student. It learns from those masks and segments mushrooms in real-time.
>
> **You** are the curator. You select images, verify quality, annotate hard cases, and decide when the student is good enough to deploy.

---
