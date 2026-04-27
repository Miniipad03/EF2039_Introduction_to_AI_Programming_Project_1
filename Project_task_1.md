Introduction to AI Programming
Project Assignment: CNN-Based
Image Classification
2026 Spring Semester
Assigned: 2026-04-02
Due: 2026-05-03
1 Project Overview
In this project, each team will design, train, and evaluate a convolutional neural network (CNN) for an image
classification task. The main goal is to build a model that can classify images from a sufficiently large dataset
with strong performance, while also demonstrating a clear understanding of data preparation, model design,
training strategy, and result analysis.
This is not intended to be a simple “run a pretrained model and report one accuracy number” assignment.
Instead, you are expected to complete the full deep learning workflow:
• choose an appropriate image classification dataset,
• preprocess and analyze the dataset,
• implement a baseline CNN model,
• improve the model using a stronger CNN architecture or better training strategies,
• evaluate the model carefully,
• and present your findings with proper evidence and discussion.
A full production-level web service is not required. However, your project must include a simple inference
demo that allows a user to test the trained model on new images.
2 Project Objectives
By completing this project, you should be able to:
• understand how CNN-based image classification systems are developed,
• compare a baseline model and an improved model fairly,
• apply training strategies such as augmentation and hyperparameter tuning,
• analyze model behavior using quantitative and qualitative results,
• and demonstrate your trained model through a lightweight inference application.
3 Dataset Requirements
You must use a publicly available image classification dataset. The dataset should be meaningful in size and
difficulty.
Your dataset must satisfy the following conditions:
• it must contain multiple classes,
• it must contain a sufficient number of images for CNN training,
• it must be appropriate for an image classification task,
• and it must be approved by the instructor if you choose a custom dataset.
1
3.1 Recommended Dataset Characteristics
A suitable dataset should typically have:
• at least 10 classes, or comparable classification difficulty,
• at least 5,000 total images,
• noticeable visual variation across images,
• and a clear train/validation/test split, or enough data to create one.
3.2 Possible Dataset Examples
You may choose one of the following, or propose another appropriate dataset:
• ImageNet
• Oxford 102 Flowers
• Food-101
• Intel Image Classification
• other public image datasets from Kaggle or similar sources
Datasets that are too easy, too small, or not appropriate for CNN-based classification may be rejected.
4 Model Requirements
Your project must include at least two CNN-based models.
4.1 Model 1: Baseline CNN
You must implement a custom baseline CNN model by yourselves. This model should include standard CNN
components such as:
• convolution layers,
• activation functions,
• pooling layers,
• fully connected layers,
• and optional regularization methods such as dropout or batch normalization.
4.2 Model 2: Improved CNN Model
You must also train and evaluate a stronger model. This may be based on a well-known CNN architecture such
as:
• ResNet
• VGG
• DenseNet
• MobileNet
• EfficientNet
You may use transfer learning or pretrained weights, but you must clearly explain:
• whether pretrained weights were used,
• whether layers were frozen or fine-tuned,
• and why this design choice was reasonable.
2
5 Required Experiments
You are expected to perform meaningful experiments rather than training only once.
Your project must include:
• one baseline experiment,
• one improved model experiment,
• and at least two attempts to improve performance.
Examples of improvement strategies include:
• data augmentation,
• learning rate tuning,
• optimizer change,
• batch size tuning,
• dropout or batch normalization,
• transfer learning,
• input image size adjustment,
• model depth or architecture comparison.
Random trial-and-error is not enough. You should make reasonable design decisions and explain why you
tried them.
6 Evaluation and Analysis
Your final report must include both numerical results and interpretation.
6.1 Training Results
For each model, include:
• training loss curve,
• validation loss curve,
• training accuracy curve,
• validation accuracy curve,
• final test accuracy.
6.2 Detailed Evaluation
You must also provide:
• a confusion matrix,
• class-wise performance analysis,
• examples of correct predictions,
• examples of failure cases,
• and a short discussion of why some classes are difficult.
3
6.3 Model Comparison
You must compare your models in terms of:
• classification accuracy,
• overfitting or underfitting behavior,
• training time,
• model complexity,
• strengths and weaknesses of each approach.
7 Inference Demo Requirement
Your project must include a simple demo showing how the trained model can be used on new images.
Acceptable examples include:
• a Python script that loads an image and prints the predicted class,
• a Jupyter notebook demo,
• a lightweight Gradio or Streamlit interface.
A full-scale deployed web service is not required. The purpose of this demo is to show that your trained model
can be used in practice.
8 Deliverables
Each team must submit the following:
8.1 Source Code
Submit all code related to:
• dataset loading,
• preprocessing,
• model implementation,
• training,
• evaluation,
• and inference demo.
8.2 Final Report
Submit a PDF report. A recommended structure is:
1. Introduction
2. Dataset Description
3. Methodology
4. Model Architectures
5. Experimental Setup
6. Results
7. Error Analysis
8. Discussion
9. Conclusion
10. References
4
8.3 Presentation
Prepare a short presentation summarizing:
• the chosen problem,
• the dataset,
• the models,
• the main results,
• and the key lessons learned.
8.4 Repository
Submit a GitHub repository link containing your code and a clear README file. The README should explain:
• project goal,
• dataset information,
• environment and dependencies,
• training instructions,
• evaluation instructions,
• and how to run the inference demo.
9 Grading Criteria
The project will be evaluated based on the following criteria:
• Dataset selection and problem setup (15%)
appropriateness of the dataset, task clarity, and difficulty level
• Model design and implementation (20%)
correctness and quality of the baseline and improved models
• Experiments and performance improvement (25%)
meaningful tuning, fair comparison, and improvement effort
• Analysis and interpretation (25%)
learning curves, confusion matrix, failure analysis, and discussion
• Code, report, and presentation quality (15%)
clarity, organization, reproducibility, and professionalism
10 Important Notes
• Simply reporting one final accuracy value is not sufficient.
• Using only a pretrained model without analysis is not sufficient.
• Training only one model is not sufficient.
• A smaller but well-analyzed project is better than a larger but poorly explained one.
• Good projects show not only performance, but also careful reasoning and interpretation.