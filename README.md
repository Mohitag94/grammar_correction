# Fine-Tune T5 LoRA for Grammar Correction on JFLEG Dataset

## 7PAM2015-0509-2024 -- Research Methods in Data Science
### Mohit Agarwal (Student ID-22031257)

## Project Overview
This project fine-tunes a T5 model to correct grammatical errors in English text. Uses JFLEG dataset for training and evaluation.

- [📓 Colab Notebook](https://colab.research.google.com/github/Mohitag94/grammar_correction/blob/main/grammar_correction_t5_lora.ipynb)

## Dataset
The assignment uses JFLEG (JHU FLuency-Extended GUG) dataset for grammatical error correction.
- Contains 1,511 sentences with 4 human-written fluency corrections each
- Split into development (754 sentences) and test sets (747 sentences)
    - development set used as training and test set used as validation & testing sets
- Focuses on fluency edits beyond just grammatical corrections
- Citation:
	Napoles, C., Sakaguchi, K., & Tetreault, J. (2017).
	JFLEG: A Fluency Corpus and Benchmark for Grammatical Error Correction

## Model Architecture
- **Base Model**: T5-small 60 million parameters
-  **Fine-tuning**: Using LoRA (Low-Rank Adaptation)

## Project Structure
```
Assignment2/Grammar_Correction
├── grammar_correction_t5_lora.ipynb	# Main training and evaluation notebook
├── README.md                   		# This file
├── Images/                     		# Directory for saving output images
├── ../Checkpoints/                     # Directory for saving training logs

```

## Configuration
The model is trained with the following configuration:
- Batch size: 4
- Number of epochs: 3
- Learning rate: 0.0001
- Warm Ratio: 0.1
- Weight Decay: 0.01
- LoRA Rank: 4
- LoRA Alpha: 8
- LoRA Dropout: 0.1
- Device: CUDA (if available) or CPU
- Random seed: 42

## Implementation Details
The implementation includes:
1. Augmented the training set through mapping each sources to it's multiple references.
2. Setup the LoRA T5 model with trainable params: 147,456 || all params: 60,654,080 || trainable%: 0.2431.
3. Evaluating every epochs on gleu, bertscore and meteor metrics.
4. Visualisation of training and validation loss along with metrics and learning rate.
5. Running test on the 10% of the validation data.
6. Inference on few sentences.

## Requirements
- datasets==3.6.0
- evaluate==0.4.4
- huggingface-hub==0.33.1
- ipykernel==6.29.5
- ipython==9.3.0
- jupyter_client==8.6.3
- jupyter_core==5.8.1
- matplotlib==3.10.3
- matplotlib-inline==0.1.7
- nltk==3.9.1
- numpy==2.3.1
- pandas==2.3.0
- pycocotools==2.0.10
- rouge_score==0.1.2
- scikit-learn==1.7.0
- seaborn==0.13.2
- textstat==0.7.7
- tokenizers==0.21.2
- torch==2.7.1
- torchaudio==2.7.1
- torchvision==0.22.1
- tqdm==4.67.1
- transformers==4.52.4
- wordcloud==1.9.4

## Usage
1. Clone the repository
2. Set up the environment with required dependencies
3. Run the `grammar_correction_t5_lora.ipynb` notebook for eda, training and evaluatating the model

## Results
The model is evaluated on the validation set using following metrics:
- **GLEU (Generalized Language Evaluation Understanding)**: Specifically designed for grammar error correction; recall oriented
	- Achieved 66.58% on test data.
- **METEOR**: Incorporates linguistic  understanding. 
	- Achieved  86.94% on test data.
- **BERT Score**: Measures semantic similarity 
	- **Precision**: 92.32% on test data.
	- **Recall**: 93.25% on test data.
	- **F1**: 92.76% on test data.


## License
This project is part of the Research Methods in Data Science course assignment.

## Acknowledgments
- JFLEG dataset creators
- Hugging Face 
- Research Methods in Data Science course instructors