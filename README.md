# Margin Perceptron
The group project of CMSC5724 2024fall
>    Li Yifei 1155215544

>    Wang Yu 1155215635

>    Tang Jiayi 1155215625

>    Fu Tianxing 1155215550

File Structure:

```
├── Perceptron.py                           # Source Code                                    
├── Dataset1.txt
├── Dataset2.txt
├── Dataset3.txt
├── CMSC5724_MarginPerceptron_Report.docx       # Project report
├── README.md                               # Project description and instructions
```


## Table of Contents
1. [Background](#background)
2. [Dataset Description](#dataset-description)
3. [Environment Requirements](#environment-requirements)
4. [Installation Instructions](#installation-instructions)
5. [Usage Instructions](#usage-instructions)
6. [Output Description](#output-description)
7. [Report](#report)
## Background
The Margin Perceptron algorithm is a machine learning algorithm used for classification tasks. It aims to find a hyperplane that maximizes the margin between classes.
## Dataset Description
Dataset format:
- The first line contains three numbers `d`, `n`, and `r`, representing the number of dimensionality, samples and radius.
- From the second line, each line represents a data point in the format: `x1,x2,...,xd,label`, where the first `d` values are coordinates, and `label` is either `1` or `-1`.
Example datasets provided:
- `2d-r16-n10000`
- `4d-r24-n10000`
- `8d-r12-n10000`
## Environment Requirements
- Python 3.x
## Usage Instructions
- Ensure dataset files are in the correct directory.
- Run the main program, specifying the list of dataset file paths.
## Output Description
During the training process, each iteration outputs the following information:
- **Current Weight Vector**: The weights at the start of the iteration.
- **Margin of Violating Sample**: The margin value of the sample violating the classification boundary.
- **Violating Sample**: Feature values of the sample that violates the classification requirement.
- **Label of Violating Sample**: The label (1 or -1) of the violating sample.
- **Updated Weights**: The updated weights after processing the violating sample.
### Output Example
Below is an example output for a single iteration:  
```
Iteration 1    
Current Weights: [0, 0]  
Margin of violating sample: 0    
Violating sample: ['-8.459748692855033', '5.621530829461882']  
Label of violating sample: -1
Weights after update: [8.459748692855033, -5.621530829461882]
```
When no violations remain, the training stops, displaying the final weights and margin:
```
Training complete - no violations.
Final Weights after 8 iterations: [4.0068648583112605, -40.90265408362018]
Final gamma: 4.0
Final weights: [4.0068648583112605, -40.90265408362018]
```
Or the maximum iterations are reached, the training stops:
```
Training stopped - maximum iterations reached.
```
## Report
The generated report includes margin analysis and result summary of the classifier.
