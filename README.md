# COMP64501 - Topics in Machine Learning
### Coursework 2026
This coursework assignment requires you to implement, train, and evaluate a machine learning model for image classification on the Fashion-MNIST dataset. You will need to define your own model architecture and training procedure, then submit a Docker container with your code and trained model weights for evaluation. After submitting your code, you will be invited to a short interview with a GTA to discuss your design choices and the development of your model.

## Prerequisites
- [Docker](https://docs.docker.com/engine/install/)
- [Python 3.12, 3.13 or 3.14](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/)

## Overview
### First time setup
1. Clone or download this repository. 
    ```bash
    cd /path/to/assignment
    ```
2. Set your 8-digit student ID number in `submission/STUDENT_ID.py`.
3. Install the required dependencies:
    ```bash
    uv sync
    ```
4. (Optional) Have a look at the data! We'll be using the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset for this assignment. If you would like to explore it yourself, visualise_dataset.ipynb is provided as a starting point.
5. (Optional) Install any additional packages you need with uv. Use `uv add --dev <package-name>` for development tools that your submitted code does not import (these are not installed during marking), and `uv add <package-name>` only for packages that your submitted code needs. For example, to add Jupyter support, you can run:
    ```bash
    uv add --dev notebook
    ```
    **IMPORTANT:** The `--dev` flag means that the package is for development tools only and is not installed during marking. **If your submitted code depends on a development package, it will fail during marking.** You have been warned!

### Defining and training your model
You will have to define a model architecture and a training procedure for this assignment, then train and evaluate it on the Fashion-MNIST dataset.
1. Implement your model in `submission/fashion_model.py`.
2. Edit the training and evaluation functions in `submission/fashion_training.py`, as needed. 
3. Perform cross-validation for hyperparameter selection and final model evaluation.

You can train your model by running the following command from the project root directory:
```bash
uv run -m submission.fashion_training
```
You may wish to modify this training call or call it from a separate script or notebook to experiment with different hyperparameters, architectures, etc.

### Testing your model
You can test your model by running the following command from the project root directory:
```bash
uv run model_calls.py
```
This will test that your model can be loaded and makes predictions in the expected format. It requires that you have already trained and saved your model weights to `submission/model_weights.pth`. We will use a script similar to this to evaluate your model during marking.

**This does not count as a cross-validation or validation script. You should implement your own validation procedure to evaluate your model's performance on a validation set during development.** 

### Preparing your code for submission
1. Use the provided Dockerfile. **Do not modify it**: we will replace it with our own copy during marking. 
2. Ensure that your trained model weights are saved to `submission/model_weights.pth`. You can do this by running your training script/notebook.
3. When you think you are ready to submit your code, run:
```bash
uv run create_submission.py
```
This will create a zip file named `STUDENT_ID.zip` in the current directory. 

Your submission zip file should contain, at minimum:
```
submission/
├── fashion_model.py           # Your model architecture
├── fashion_training.py        # Your training code
├── model_weights.pth          # Pre-trained weights
├── STUDENT_ID.py              # Your student ID
├── engine.py                  # Training helpers - imported by fashion_training.py
└── __init__.py                # (required) Empty file
model_calls.py                 # (DO NOT MODIFY - will be replaced)
utils.py                       # (DO NOT MODIFY - will be replaced)
Dockerfile                     # (DO NOT MODIFY - will be replaced)
.dockerignore                  # (optional)
pyproject.toml                 # Project configuration
uv.lock                        # Dependency lock file
```
You should include additional files as needed in the submission/ directory, such as additional modules or scripts you use for hyperparameter selection, data preprocessing, etc.. Please avoid including jupyter notebooks or large data files in your submission.

**IMPORTANT:** If your .zip file is larger than 2MB, you will receive a warning. Please check that your submission only contains the files required for submission and does not include any unnecessary files (e.g., Jupyter notebooks, large datasets, etc.). Large submissions slow down the marking process, and you **will** be penalised if it is too large.

### Final checks
Before submitting, it is strongly recommended to test your submission using Docker:
1. Unzip the created submission file into a new directory:
   ```bash
   unzip STUDENT_ID.zip -d .
   ```
2. Change into the unzipped directory:
   ```bash
   cd STUDENT_ID
   ```
3. Build the Docker image:
   ```bash
    docker build -t test .
    ```
4. Run the Docker container with **no** network access. This is how your submission is run during marking, so make sure this passes.
   ```bash
    docker run --rm --network=none test
   ```
4. Run the Docker container **with** network access, so that it can download the public test set and check that your model makes predictions:
   ```bash
    docker run --rm test
   ```
If all tests pass, your submission is ready to be uploaded to the course portal on Canvas.

## Understanding the evaluation process

During marking, the following happens:

1. **File replacement**: `model_calls.py` and `utils.py` are replaced with marking versions (**important:** any changes you make to these files will not be saved!)
2. **Docker build**: Our copy of the provided Dockerfile is used to build a container
3. **Container execution**: When build is finished, the container runs `model_calls.py` without network access
4. **Model loading**: Your architecture is instantiated and weights that you submitted are loaded
5. **Integrity checks**:
   - Model architecture works (forward/backward pass)
   - Preprocessing transforms are valid: training transforms can be any standard torchvision transform (no lambda functions, sorry), but evaluation transforms must be deterministic and come from the allowed list in utils.py (this helps ensure that we can reproduce your results)
   - Training function works and returns valid format
6. **Evaluation**: Model is evaluated on the test set
7. **Results**: Accuracy, parameter count, and results from checks are recorded

**Output format** during marking (parsed automatically) looks like this:
```
STUDENT_ID: 10045678
ACCURACY_CHECK: PASSED
PARAMETERS: 45820
TRAINING_CHECK: PASSED
```

## Submission

Submit the .zip file through the assignment portal on Canvas by the deadline of **14:00 November 17, 2026, Tuesday**.

## Marking criteria
The assessed coursework is worth 20 marks and is 20% of the overall module mark. It will be assessed according to the following criteria:
### Code and model performance (11 marks):
- **Code functionality (9 marks):**
    - Model architecture is valid and can be instantiated 
    - Training function works and returns valid weights 
    - Model achieves **at least 88% accuracy** on OUR test set (not the provided one, but in the same format) 
    - Model efficiency: <= 100,000 parameters 
    - Submission size is <= 2MB (contact me if you think you have a good reason to exceed this limit)
    - I do not have to put any extra effort into debugging your submission (renaming files, fixing file structure, etc.).
- **Model performance (2 marks):**
    - Marks for accuracy: the top submissions by test set accuracy (e.g. above the median, or in the top 20% [^*]) will receive up to two marks.
    - Marks for parameter efficiency: the most efficient submissions that still meet the accuracy threshold will receive one mark (e.g. the 30% of submissions with the fewest parameters [^*]).
    - Note that these marks are not cumulative; the maximum you can receive for model performance is 2 marks.
### Interview (9 marks):
During the interview, a GTA will ask about your model development process and provide you with the opportunity to justify your design decisions. They will ask three questions about your submitted code for a total of three marks each. **No laptops will be allowed during the interview.** We will not ask you to complete any derivations or write any pseudocode during this interview; it is designed to assess your understanding of the development process and the code you have submitted.

[^*]: I know I'll get a lot of questions about this. I reserve the right to ease these criteria as we receive more submissions. I will do my best to ensure that the bonus marks are allocated fairly and there's a justifiable cutoff point based on the distribution of submissions we receive, not an arbitrary threshold that can be attained or missed through e.g. stochasticity in the test set. 

### AI policy
The AI policy for the code is **AI-permitted.** You are free to use AI coding assistants to help you develop this coursework. However, no laptops, notes, or tools are permitted during the interview (**AI-prohibited**). The interview is designed so that you may demonstrate an understanding of the code that you submit.

### Late submissions
We follow Department's guidelines about late submissions, i.e., a deduction of 10% of the mark each 24 hours the work is late after the deadline. NO late submission will be marked one week after the deadline. Please read [this link](https://documents.manchester.ac.uk/DocuInfo.aspx?DocID=29825). 

### Use of unfair means 
**Any form of unfair means is treated as a serious academic offence and action may be taken under the Discipline Regulations.** Please carefully read [what constitutes Unfair Means](https://documents.manchester.ac.uk/display.aspx?DocID=639) if not sure. If you still have questions, please ask your Personal tutor or the Lecturers. If you use AI (e.g., ChatGPT) to help you with any part of your submission, you must clearly state so in your submission comments, including details of how you used it. Failure to do so will (1) lose you a mark and (2) be treated as an academic offence.
