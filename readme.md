# Exploring the Effects of Data Augmentation on Dialogue Reasoning. 


To enhance reasoning, datasets like MuTual emphasize complex reasoning challenges. Our approach incorporates generative models like Llama 2 to create synthetic labels, with the aim of evaluating their impact on dialogue system performance.

**Data**

To obtain the data, execute git clone https://github.com/Nealcly/MuTual and place the data folder in the current directory.

**Results**
To print directly the results of the 2 tables, run python -u error_analysis.py 


**RoBERTa Experiments**

*RoBERTa binary classification baseline*\
python -u train.py

*Fine-Tuned Llama Integration:* Merged the original dataset with all generated labels by the Llama model fine-tuned on a subset of original samples, as positive examples.\
python -u train.py --augment final_finetuned.pkl

*Threshold-based Fine-Tuned Llama Integration:* Employed the original dataset with generated labels by the Llama model fine-tuned on a subset of original samples, classified as either positive or negative based on a set threshold.\
python -u train.py --augment final_finetuned.pkl --sim

*True label ablation:* Repeat the authentic true label for each dialogue \
python -u train_repeat.py --learning_rate 2e-05 --repeat_type gold --repeat_pickle manually_final_finetuned.pkl

*Ablation including Similarity Filtering:* In the case of the threshold-included setup, the true label is upsampled only if the generated utterance is classified as positive. Otherwise, a random negative label out of the given options for a given dialogue is chosen to be replicated.\
python -u train_repeat.py --repeat_type sim --repeat_pickle sim_final_finetuned.pkl

**LLama**

To request access for LLama, you need to follow the following steps:
1. Request access from Meta: Visit the Meta AI website to request access to LLama at https://ai.meta.com/resources/models-and-libraries/llama-downloads/.

2. Create a Hugging Face account: Go to the Hugging Face website and sign up for an account at https://huggingface.co/signup.

3. Request access to the LLama model: Once you have a Hugging Face account, request access to the LLama model by visiting the following URL: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf.

4. Create your own Hugging Face API token: After requesting access to the LLama model, you will need to create your own Hugging Face API token. Follow the instructions provided by Hugging Face to generate your token.

Then run:
python -u train_llama.py --do_train --use_context --bits 8 --top_p 0.95 --hf_token ADD_YOUR_TOKEN_HERE



