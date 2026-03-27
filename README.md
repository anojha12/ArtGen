# ArtGenProject

With the rapid advancement of image generation models such as DALL·E and MidJourney, distinguishing between AI-generated and human-created images has become increasingly important. This project focuses on building robust models to classify images based on their origin.

Despite significant progress in detection techniques—including deep learning models, frequency-domain analysis, and input manipulation—generalization and accuracy remain major challenges. This work aims to explore and improve detection performance across diverse datasets.

Objectives:
Develop models to classify images as:
AI-generated
Human-created
Address dataset imbalance issues
Improve generalization across styles and domains
Compare different detection paradigms

Datasets:
1. AI Recognition Dataset
~17,900 AI-generated images
Sources include:
DALL·E
MidJourney
~4,000 human-painted images collected from the internet
2. Art Dataset
~500,000 artwork images
Categorized by genre
Used to enrich diversity of human-created samples
Data Balancing

Methodology:
This project explores three major approaches.

1. Frequency-Domain Forensics
Detect artifacts in Fourier space
AI-generated images often exhibit:
Repetitive patterns
Abnormal high-frequency signals
2. Vision-Language & Multimodal Models
Combine image features with semantic understanding
Use pretrained models (e.g., CLIP-like architectures)
3. Statistical / Entropy-Based Methods
Analyze:
Pixel distribution
Noise patterns
Compression artifacts
Hypothesis: AI images differ statistically from natural images
