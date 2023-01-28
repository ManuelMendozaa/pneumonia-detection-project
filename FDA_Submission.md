# FDA  Submission

**Your Name:** Manuel Mendoza.

**Name of your Device:** Deep Learning Model for Pneumonia Detection.

## Algorithm Description

### 1. General Information

**Intended Use Statement:**
This algorithm is intended only to assists radiologists when diagnosing pneumonia on Chest X-Rays. It should not be used as a substitute for radiologists or any clinician in the clinical setting.

**Indications for Use:**
This algorithm could be applied in patients of ages from 1 to 95 years old, with medical history indicating pneumonia. The X-Ray study position must be either PA or AP.

**Device Limitations:**
Since the algorithm works through an architecture of convolutional neural networks, GPU and RAM resources must be presented in the desire device.
Some of the following preexisting or suspected conditions may not disturb in the algorithm output: Infiltration, effusion, atelectasis, nodules, mass, pneumothorax, consolidations, pleural thickening, cardiomegaly, emphysema, edema, fibrosis or hernias. Any other condition could influence in the model's prediction.

**Clinical Impact of Performance:**
Four possible cases can be taken into consideration at the momento of evaluating the impact of the algorithm's performance by its own:
- True Positive: *Average impact* since any further actions and procedures would follow the same path as if a clinician had examined the study.
- True Negative: *Low impact* since no actions or procedures should correctly take place for the diagnosis of pneumonia.
- False Positive: *Medium impact* since the patient would be involved in unnecessary procedure, but their health would not at stake.
- False Negative: *High impact* since the patient would not take the necessary procedures, risking their condition.

With these four scenarios in mind, the algorithm's prediction should only be used to assist the radiologist's final decision.

### 2. Algorithm Design and Function

![flowchart](images/flowchart.png)

**DICOM Checking Steps:**


**Preprocessing Steps:**

**CNN Architecture:**


### 3. Algorithm Training

**Parameters:**
* Types of augmentation used during training
* Batch size
* Optimizer learning rate
* Layers of pre-existing architecture that were frozen
* Layers of pre-existing architecture that were fine-tuned
* Layers added to pre-existing architecture

<< Insert algorithm training performance visualization >>

<< Insert P-R curve >>

**Final Threshold and Explanation:**

### 4. Databases
 (For the below, include visualizations as they are useful and relevant)

**Description of Training Dataset:**


**Description of Validation Dataset:**


### 5. Ground Truth



### 6. FDA Validation Plan

**Patient Population Description for FDA Validation Dataset:**

**Ground Truth Acquisition Methodology:**

**Algorithm Performance Standard:**
