# Capstone - Convolutional Neural Network by Ian Ryan
<p>Welcome to My Fall 2024 Capstone Project Advised by Dr. Soltys.</p>

<p>The objective of this semester-long project was to create a Convolutional Neural Network (CNN) using PyTorch by scratch, rather than using pre-trained models like ResNet-18. After developing an acceptable classification model, the next goal was to adapt it into an object detection model by implementing YOLOv8.</p>

<p>Although my primary interest was detecting humans, I chose to create a multiclass model to avoid building a simple perceptron. To up the ante, I also made the dataset that the model was trained on an imbalanced dataset. Developing my own model presented a challenging learning opportunity and helped strengthen my understanding of CNN fundamentals for computer vision, a field in which I aspire to build a career.</p>

<p>This repository contains the majority of my work, including various iterations of fine-tuning, visualizations, the CNN architecture schematic, datasets used, comprehensive Jupyter Notebooks, my capstone poster, final capstone pitch, literature citations, and more. The Jupyter Notebooks feature detailed indexes for easy navigation through different sections of the programs.</p>

## CNN Architecture

![CNN Architecture](https://github.com/ianhenryryan/capstone/blob/main/Cap/images/architecture320.jpg?raw=true)
<p>Below is the table of contents for the Classification Jupyter Notebook in the notebooks directory specifically the capstone subdirectory. (if not already, the Object Detection CNN with YOLOv8 Jupyter Notebook will be uploaded soon)</p>

<h1>Convolutional Neural Network - Computer Vision (Classification)</h1>

<h2>Index</h2>
<ul>
  <li><strong>Table of Contents</strong>
    <ul>
      <li><strong>Libraries & Imports</strong>
        <ul>
          <li>Libraries</li>
          <li>Imports</li>
          <li>CUDA GPU Availability</li>
        </ul>
      </li>
      <li><strong>Download & Load Datasets</strong>
        <ul>
          <li>First Dataset Resource</li>
          <li>Second Dataset Resource</li>
          <li>Functions</li>
          <li>Path Datasets
            <ul>
              <li>First Dataset</li>
              <li>Second Dataset</li>
            </ul>
          </li>
        </ul>
      </li>
      <li><strong>Pre-Processing The Datasets</strong>
        <ul>
          <li>Combine Datasets</li>
          <li>Save Combo Set</li>
          <li>Redistribute Images Dynamically Function</li>
          <li>Pathing (Jump here after Libraries & Imports if using own dataset and not combining into imbalanced set.)
            <ul>
              <li>Normalize File Extensions</li>
              <li>Combo Transforms Train Valid Test</li>
              <li>Extensive Pathing Checking</li>
            </ul>
          </li>
          <li>Transforms & Loaders</li>
        </ul>
      </li>
      <li><strong>Classification Or Object Detection Model?</strong>
        <ul>
          <li>Classification Task</li>
          <li>Regression Task</li>
        </ul>
      </li>
      <li><strong>CNN Model</strong>
        <ul>
          <li>Define Class Weights</li>
          <li>Define CNN Model Architecture</li>
          <li>Weights & Biases</li>
          <li>Hyperparameters</li>
          <li>Class Confirmation in Training Data</li>
          <li>Profile Memory Usage</li>
          <li>Training Prep</li>
          <li>Weight Initialization</li>
        </ul>
      </li>
      <li><strong>Training CNN Model</strong>
        <ul>
          <li>Test Accuracy of Convolutional Neural Network</li>
        </ul>
      </li>
      <li><strong>Visualizations</strong>
        <ul>
          <li>Summary - Everything</li>
          <li>Model Performance Function</li>
          <li>Training & Validation Loss Plots</li>
          <li>Confusion Matrices</li>
          <li>Feature Maps</li>
          <li>Kernel Visualizations</li>
          <li>Gradient Visualizations</li>
          <li>CAM / Grad-CAM</li>
          <li>Training/Validation Curves</li>
          <li>Explainability Tools</li>
        </ul>
      </li>
      <li><strong>Acknowledgements</strong></li>
      <li><strong>Literature Cited</strong></li>
      <li><strong>Environment</strong></li>
      <li><strong>Recommended Resources</strong></li>
      <li><strong>Creator Information</strong></li>
      <li><strong>Permission</strong></li>
    </ul>
  </li>
</ul>

# **Acknowledgements**
<p>I would like to express my gratitude to <strong>Dr. Michael Soltys</strong> and <strong>Dr. William Barber</strong> for sparking my interest in Machine Learning and Artificial Intelligence. Over the past two years at <em>California State University Channel Islands (CSUCI)</em>, it has been a pleasure taking multiple courses with both professors, each bringing their own perspectives and experiences. They both were able to articulate complex information in a digestible way effortlessly.</p>

<h3>Dr. William Barber</h3>
<p>
Dr. Barber, with his background in <strong>Physics</strong> and a distinguished career in <strong>Medical Imaging and Research</strong>, provided a scientific and mathematical foundation that enhanced my understanding of computational models, data analysis, and data extraction. His extensive industry experience includes serving as <strong>Director of Medical Imaging at Rapiscan Systems</strong>. His engaging teaching style and passion for image processing techniques and pattern recognition concepts inspired me to explore this field further.
</p>

<h3>Dr. Michael Soltys</h3>
<p>
Dr. Soltys, with his comprehensive background in <strong>Algorithms, Machine Learning, and Cloud Computing</strong>, fostered my interest in pursuing a Computer Vision Capstone Project. As an accomplished author of two books and more than 60 published research papers, Dr. Soltys brings a wealth of in-depth knowledge and real-world applications to his teaching. His courses and recommendations proved to be instrumental during the development of my <strong>Convolutional Neural Network (CNN)</strong> Capstone Project.
</p>

<h3>Bayne H. Ryan</h3>
<p>
Bayne, my dear nephew. You arrived the first week of December, right in the middle of preparations for my Capstone Showcase. I truly admire your impeccable sense of urgency to stop being a submarine and surface just in time to appreciate my Convolutional Neural Network (CNN) Capstone project. Your surprise guest appearance was undoubtedly the highlight of my year. I cannot wait to forcibly teach you calculus when you reach the prestigious age of juice boxes and nap time (also known as four years old).
</p>

# **Literature Cited**
> Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv preprint arXiv:1901.08688.

> Brownlee, J. How to Develop Convolutional Neural Network Models for Time Series Forecasting. Machine Learning Mastery. Available at: https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/.

> Dougherty, G. Pattern Recognition and Classification: An Introduction. Springer, 2012.

> Eliot, D. Deep Learning with PyTorch Step-by-Step: A Beginner's Guide, Volume I: Fundamentals. Self-published, 2020.

> Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. Available at: https://github.com/ultralytics/ultralytics.

# **Environment**
- **AWS SageMaker Jupyter Notebook**
  - **Notebook Instance Type:** ml.g4dn.2xlarge
  - **Processor:** NVIDIA T4 Tensor Core GPU with 16 GB GDDR6 VRAM
  - **Memory:** 32 GB System RAM
  - **Storage:** 5 GB EBS Volume
  - **Operating System:** Amazon Linux 2 with Jupyter Lab 3 (notebook-al2-v2)
  - **Lifecycle Configuration:** None
  - **Elastic Inference:** Not Enabled
  - **Minimum IMDS Version:** 2

# **Recommended Resources**
<p>Courses for Computer Science Students at <em>California State University Channel Islands (CSUCI)</em> Interested in Machine Learning & AI</p>

<h3><u>Channel Island Specific Courses:</u></h3>
<ul>
  <li><strong>COMP 345</strong> - Digital Image Processing (Fall Course)</li>
  <li><strong>COMP 354</strong> - Analysis of Algorithms</li>
  <li><strong>COMP 445</strong> - Image Analysis and Pattern Recognition (Spring Course)</li>
  <li><strong>COMP 454</strong> - Automata, Languages & Computation</li>
  <li><strong>COMP 469</strong> - Intro to Artificial Intelligence / Machine Learning (Fall Course)</li>
  <li><strong>COMP 491</strong> - Capstone Prep (Specifically with Dr. Soltys)</li>
  <li><strong>COMP 499</strong> - Capstone (Specifically with Dr. Soltys)</li>
</ul>

<p><em>Correlating Instructor to Specific Coursess:</em></p>
<ul>
  <li>COMP 345 & 445 are taught by Dr. Barber</li>
  <li>COMP 354, COMP 454, & COMP 469 are taught by Dr. Soltys</li>
</ul>

<h3><u>AWS Courses</u> (if offered):</h3>
<ul>
  <li>AWS MLU Machine Learning through Application (<strong>One Half of COMP 469</strong> with <strong>Soltys</strong> and it gives you a <strong>certificate</strong>)</li>
  <li>AWS MLU Application of Deep Learning to Text and Image Data (<strong>One Half of COMP 469</strong> with <strong>Soltys</strong> and it gives you a <strong>certificate</strong>)</li>
  <li>AWS DeepRacer Student</li>
</ul>

<h3><u>LinkedIn Learning Courses</u> (if offered):</h3>
<p>If a College/University student, it is worth seeing what resources your institution provides at your disposal. For example, CSUCI offers LinkedIn Learning and access to the O'Reilly website.</p>
<ul>
  <li>
    <a href="https://www.linkedin.com/learning/deep-learning-and-generative-ai-data-prep-analysis-and-visualization-with-python/leverage-generative-ai-for-analytics-and-insights?u=37164436" target="_blank">
      Deep Learning and Generative AI: Data Prep, Analysis, and Visualization with Python Leverage Generative AI for Analytics and Insights by Gwendolyn Stripling
    </a>
  </li>
  <li>
    <a href="https://www.linkedin.com/learning/deep-learning-image-recognition-24393297/learning-image-recognition?u=37164436" target="_blank">
      Deep Learning: Image Recognition Learning image recognition by Isil Berkun
    </a>
  </li>
  <li>
    <a href="https://www.linkedin.com/learning/advanced-ai-transformers-for-computer-vision/transformers-for-computer-vision?u=37164436" target="_blank">
      Advanced AI: Transformers for Computer Vision by Jonathan Fernandes
    </a>
  </li>
  <li>
    <a href="https://www.linkedin.com/learning/building-computer-vision-applications-with-python/building-computer-vision-applications-with-python?u=37164436" target="_blank">
      Building Computer Vision Applications with Python by Eduardo Corpeño
    </a>
  </li>
  <li>
    <a href="https://www.linkedin.com/learning/applied-machine-learning-algorithms-23750732/applied-machine-learning-algorithms?u=37164436" target="_blank">
      Applied Machine Learning: Algorithms by Matt Harrison
    </a>
  </li>
  <li>
    <a href="https://www.linkedin.com/learning/building-deep-learning-applications-with-keras/reshaping-the-world-with-deep-learning?u=37164436" target="_blank">
      Building Deep Learning Applications with Keras by Isil Berkun
    </a>
  </li>
</ul>

<h3><u>Helpful Resources:</u></h3>
<p>Resources that I found useful while working on a Computer Vision project and learning about Machine Learning & AI.</p>

<ul>
  <li>
    <a href="https://greenteapress.com/thinkpython2/thinkpython2.pdf" target="_blank">
      Think Python How to Think Like a Computer Scientist 2nd Edition, Version 2.4.0 by Allen Downey
    </a>
  </li>
    <li>
    <a href="https://github.com/dvgodoy/PyTorchStepByStep/blob/master/README.md" target="_blank">
      Deep Learning with PyTorch Step-by-Step: A Beginner's Guide: Volume I: Fundamentals by Daniel Voigt Godoy
    </a>
  </li>

  <li>
    <a href="https://link.springer.com/book/10.1007/978-1-4614-5323-9" target="_blank">
      Pattern Recognition and Classification An Introduction (Springer, 2012) by Geoff Dougherty
    </a>
  </li>

  <li>
    <a href="https://www.amazon.com/author/msoltys" target="_blank">
      An Introduction to the Analysis of Algorithms (3rd Ed, 2018) by Michael Soltys
    </a>
  </li>

  <li>
    <a href="https://docs.ultralytics.com/models/yolov8/" target="_blank">
      Ultralytics YOLOv8 Documentation
    </a>
  </li>

  <li>
    <a href="https://shisrrj.com/paper/SHISRRJ247267.pdf" target="_blank">
      Object Detection and Localization with YOLOv3 by B. Rupadevi and J. Pallavi
    </a>
  </li>

  <li>
    <a href="https://www.amazon.com/Digital-Image-Processing-Medical-Applications/dp/0521860857" target="_blank">
      Digital Image Processing for Medical Applications by Geoff Dougherty
    </a>
  </li>
</ul>

# **Creator Information**
No Guarantee that I will regularly upload things but if I do they would be at one of these locations.

- **GitHub:** 
  - https://github.com/ianhenryryan
- **LinkedIn:**
  - https://linkedin.com/in/ianhenryryan/
- **Websites:**
  - http://ianhryan.com/
  - http://swolegreekgod.com/
- **Kaggle:**
  - https://kaggle.com/ianryan
- **Hugging Face:**
  - https://huggingface.co/Ianryan
 
# **Permission**
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<p>Students, educators, and anyone else keen on learning about Convolutional Neural Networks (CNNs), Computer Vision, Supervised Learning Algorithms, Machine Learning, Deep Learning, AI, or related fields are welcome to use this notebook in any capacity as a learning resource.</p>

<p>This notebook is part of my Fall 2024 Capstone Project for my Bachelor’s degree in Computer Science at California State University Channel Islands. I structured the content to be approachable and digestible, reflecting my own learning journey in AI and Machine Learning.</p>

<p>I hope this notebook can be of use to those exploring similar topics. This specific Jupyter Notebook focuses on Image Classification and demonstrates combining two datasets to create a class imbalance for training purposes. A separate notebook dedicated to Object Detection will be available soon (if not already).</p>

<h3><Strong>Important Note on Datasets:</Strong></h3>
<p>The datasets used in this project are not my property. Credit is given to the original dataset creators, and their links are provided within the notebook in the dataset section at the beginning.</p>

<p>I understand that grasping the fundamentals of CNNs and related AI concepts can be overwhelming at first. My goal is to make these topics more accessible through these notebooks.</p>

<h3>Permissions:</h3>
<p>You are free to download, use, edit, and reference the notebooks, Python code, and Markdown content. I aim for accuracy in the explanations provided, though I acknowledge that scientific understanding is always evolving. I welcome constructive feedback and corrections.</p>

<p>This project is intended as a learning resource.</p>

<p><strong>&mdash; Ian Ryan</strong></p>

</body>
</html>
