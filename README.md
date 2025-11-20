# COMP64501 - Topics in Machine Learning
### Coursework 2025
This coursework assignment requires you to implement, train, and evaluate a machine learning model for image classification on the Fashion-MNIST dataset. You will need to define your own model architecture and training procedure, then submit a Docker container with your code and trained model weights for evaluation. Along with your code, you will submit a brief pdf report discussing your design choices and the performance of your model.

## Prerequisites
- [Docker](https://docs.docker.com/engine/install/)
- [Python 3.11+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/)

## Overview
### First time setup
1. Clone or download this repository. 
    ```bash
    cd /path/to/assignment
    ```
2. Set your student ID in `STUDENT_ID.py`.
3. Install the required dependencies:
    ```bash
    uv sync
    ```
4. (Optional) Have a look at the data! We'll be using the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset for this assignment. If you would like to explore it yourself, visualise_dataset.ipynb is provided as a starting point.
5. (Optional) Install any additional dependencies you may need for development by using `uv add <package-name>`. For example, to add Jupyter support, you can run:
    ```bash
    uv add notebook ipykernel matplotlib
    ```

### Defining and training your model
You will have to define a model architecture and a training procedure for this assignment, then train and evaluate it on the Fashion-MNIST dataset.
1. Implement your model in `submission/fashion_model.py`. 
2. Edit the training and evaluation functions in `submission/fashion_training.py`, as needed.
3. Perform cross-evaluation for hyperparameter selection and final model evaluation.

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

**This does not count as a cross-evaluation or validation script. You should implement your own validation procedure to evaluate your model's performance on a validation set during development.** 

### Preparing your code for submission
1. Create a Dockerfile, or modify the provided one, to set up the environment needed to run your code. 
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
├── engine.py                  # (optional) Training helpers
└── __init__.py                # (required) Empty file
model_calls.py                 # (DO NOT MODIFY - will be replaced)
utils.py                       # (DO NOT MODIFY - will be replaced)
Dockerfile                     # Your Docker configuration
README.md                      # This file
pyproject.toml                 # Project configuration
```
You should include additional files as needed in the submission/ directory, such as additional modules or scripts you use for hyperparameter selection, data preprocessing, etc.. Please avoid including jupyter notebooks or large data files in your submission.

In addition to this .zip file, you must also submit a brief pdf report discussing your design choices and the performance of your model. This should be typeset in [LaTeX](https://www.latex-project.org/) or [Markdown](https://www.markdownguide.org/) (converted to PDF). See this [discussion thread](https://canvas.manchester.ac.uk/courses/45977/discussion_topics/738020?entry_id=40707) for more details. 

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
4. Run the Docker container with no network access:
   ```bash
    docker run --rm --network=none test
   ```
If all tests pass, your submission is ready to be uploaded to the course portal on Canvas.

## Understanding the evaluation process

During marking, the following happens:

1. **File replacement**: `model_calls.py` and `utils.py` are replaced with marking versions (**important:** any changes you make to these files will not be saved!)
2. **Docker build**: Your Dockerfile is used to build a container
3. **Container execution**: When build is finished, the container runs `model_calls.py`
4. **Model loading**: Your architecture is instantiated and weights that you submitted are loaded
5. **Integrity checks**:
   - Model architecture works (forward/backward pass)
   - Prepocessing transforms are valid (no lambda functions, sorry) and deterministic in eval mode
   - Training function works and returns valid format
6. **Evaluation**: Model is evaluated on the test set
7. **Results**: Accuracy, parameter count, and results from checks are recorded

**Output format** (parsed automatically) looks like this:
```
STUDENT_ID: 10045678
ACCURACY: 0.872300
PARAMETERS: 45820
TRAINING_CHECK: PASSED
```

## Submission

Submit the .zip file and your pdf report through the assignment portal on Canvas by the deadline of **14:00 November 27, 2025, Thursday**.

## Marking criteria
The assessed coursework is worth 20 marks and is 20% of the overall module mark. It will be assessed according to the following criteria:
### Code and model performance (10 marks):
- **Code functionality (7 marks):**
    - Model architecture is valid and can be instantiated 
    - Training function works and returns valid weights 
    - Model achieves **at least 88% accuracy** on OUR test set (not the provided one, but in the same format) 
    - Model efficiency: <= 100,000 parameters 
- **Code quality (1 mark):**
    - Code is well-structured, readable, and appropriately commented.
- **Model performance bonus (2 marks):**
    - Additional marks for accuracy improvements: the top submissions by accuracy (e.g. top 50th percentile, top 20th percentile by test set performance [^*]) will receive up to two additional bonus marks.
    - Additional marks for parameter efficiency: the most efficient submissions that still meet the accuracy threshold will receive one bonus mark (e.g. bottom 30th percentile of parameters [^*]).
    - Note that these bonus marks are not cumulative; the maximum bonus you can receive is 2 marks.
### Model design and report (10 marks):
The report should describe your model development process, justify your design decisions, and present your results. It will be marked on three components:
- **Model architecture design (3 marks):**
    - Description of your model architecture and justification for your design choices (e.g. why did you choose convolutional layers vs fully connected, how did you determine the number of layers/units, etc.).
    - Parameter efficiency considerations: How did you balance model capacity with the parameter constraint?
- **Training and hyperparameter selection (4 marks):**
    - Description of your training procedure, including data preprocessing and augmentation (if any).
    - Clear implementation and explanation of your cross-validation strategy. How did you ensure your model generalises well to unseen data?
    - What hyperparameters did you tune (e.g. learning rate, batch size, number of epochs, etc.) and how did you make your final selections?
- **Reporting of results (3 marks):**
    - Clear presentation of training and validation results, including relevant plots (e.g. training/validation accuracy and loss curves).
    - Discussion of model performance, challenges faced, and potential improvements. 
    - All plots and tables should be clearly labeled and referenced in the text. All plots should have appropriate axis labels, legends, and titles.

Word count for report: max. 1000 words, excluding figures, tables, captions, and citations. **Please include a word count in your submission.** Figures and tables are strongly encouraged. Please cite all relevant sources and references. If you use any external code or libraries beyond those provided, please cite them appropriately.

[^*]: I know I'll get a lot of questions about this. As this is the first year we're doing this particular assignment, I reserve the right to ease these criteria as we receive more submissions. I will do my best to ensure that the bonus marks are allocated fairly and there's a justifiable cutoff point based on the distribution of submissions we receive, not an arbitrary threshold that can be attained or missed through e.g. stochasticity in the test set. 

### Late submissions
We follow Department's guidelines about late submissions, i.e., a deduction of 10% of the mark each 24 hours the work is late after the deadline. NO late submission will be marked one week after the deadline. Please read [this link](https://documents.manchester.ac.uk/DocuInfo.aspx?DocID=29825). 

### Use of unfair means 
**Any form of unfair means is treated as a serious academic offence and action may be taken under the Discipline Regulations.** Please carefully read [what constitutes Unfair Means](https://documents.manchester.ac.uk/display.aspx?DocID=639) if not sure. If you still have questions, please ask your Personal tutor or the Lecturers. If you use an LLM (e.g., ChatGPT) to help you with any part of your submission, you must clearly state so in your report, including details of how you used it. Failure to do so will be treated as an academic offence.
