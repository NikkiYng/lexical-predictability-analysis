import os
import requests
from bs4 import BeautifulSoup
import random
from openai import OpenAI
import pickle
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd
from scipy.stats import norm
#from pyprocessmacro import Process
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


''' 1. Data collection and preprocessing'''

# 1.1 Scraping 

url = "https://www.gutenberg.org/files/1228/1228-h/1228-h.htm"
response = requests.get(url)
html_content = response.text 
soup = BeautifulSoup(html_content, 'html.parser') # Parse the content and turn it to string
text_content = soup.get_text() # Extract the text content


# 1.2 Extract original text for shuffling

start_sentence = "Before entering on the subject of this chapter,"
end_sentence = "CHAPTER IV."
start_index = text_content.find(start_sentence)
end_index = text_content.find(end_sentence)
extracted_text = text_content[start_index:end_index] 
#print(extracted_text)
words = (extracted_text.split())[:60] # Extract the first 60 words
#original_text = " ".join(words)
#original_text


# 1.3 Create speech with disorder

disorder_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Levels of disorder
num_samples = 100
sample_length = len(words)
random.seed(66) 
shuffled_samples = {level: [] for level in disorder_levels}
for level in disorder_levels:
    for _ in range(num_samples):
        num_shuffled = int(level * sample_length)
        indices = list(range(sample_length)) # Create a list of indices
        random.shuffle(indices) # Shuffle the indices
        shuffled_indices = indices[:num_shuffled] + sorted(indices[num_shuffled:]) # Combine shuffled and unshuffled indices
        shuffled_text = " ".join([words[i] for i in shuffled_indices]) 
        shuffled_samples[level].append(shuffled_text)

# Check if all samples are uiniuqe
for level in disorder_levels:
    unique_samples = set(shuffled_samples[level])
    if len(unique_samples) == num_samples:
        print(f"All samples at disorder level {level*100:.0f}% are unique.")
    else:
        print(f"Not all samples at disorder level {level*100:.0f}% are unique. {len(unique_samples)} unique out of {num_samples}.")

# Read the first 3 shuffled samples
for level in disorder_levels:
    print(f"Disorder Level: {int(level*100)}%")
    for sample_idx in range(3):
        print(f"Sample {sample_idx + 1}:\n{shuffled_samples[level][sample_idx]}\n")



''' 2. Get predictability scores using gpt'''

# 2.1 Set up the OpenAI API key

os.environ['OPENAI_API_KEY'] = 'sk-proj-Yzk6roJqCe3OWogIliDy-the rest of the key'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# 2.2 Function to get log probabilities

def get_predictability_scores(text, max_context_length):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    words = text.split()
    log_probs = []
    #perplexities = [] # Not used for now
    for i in range(1, len(words)):
        if i > max_context_length:
            break
        try:
            context = ' '.join(words[:i]) 
            next_word = words[i]
            #print(f"Context length: {i}, Next Word: {next_word}")
            response = client.completions.create(
                model="babbage-002",
                prompt=context,
                max_tokens=1,
                logprobs=80 # Get log probabilities for the top 80 words
            )
            # Convert response to dictionary
            response_dict = response.model_dump() 
            word_logprobs = response_dict['choices'][0]['logprobs']['top_logprobs'][0]
            # Normalize the actual next word and predictions for comparison
            next_word_normalized = next_word.strip().lower() 
            normalized_logprobs = {key.strip().lower(): value for key, value in word_logprobs.items()} 
            if next_word_normalized in normalized_logprobs:
                log_prob = normalized_logprobs[next_word_normalized]
                log_probs.append(log_prob)
                #perplexity = np.exp(-log_prob)
                #perplexities.append(perplexity)
            else:
                log_prob = min(normalized_logprobs.values())  # Take the lowest log prob if the word is not in predictions
                log_probs.append(log_prob)
                #perplexity = np.exp(-log_prob)
                #perplexities.append(perplexity)
            # Print the log prob and perplexity
            #print(f"Log Probability: {log_prob}")
        except Exception as e:
            print(f"An error occurred: {e}")
    # Check if log probs length matches the context length
    if len(log_probs) != max_context_length:
        print(f"Warning: Generated {len(log_probs)} log_probs, expected {max_context_length}") 
    return log_probs

results = {level: [] for level in disorder_levels} 
completed_samples = {level: 0 for level in disorder_levels} 


# 2.3 Update existing progress if it gets interrupted

if os.path.exists('results.pkl'):
    with open('results.pkl', 'rb') as file:
        saved_data = pickle.load(file)
        results = saved_data['results']
        completed_samples = saved_data['completed_samples']
    print("Loaded existing progress from results.pkl")

# 2.4 Get predictability scores 

for level in disorder_levels:
    print(f"Disorder Level: {int(level*100)}%")
    start_idx = completed_samples[level]
    for idx, shuffled_text in enumerate(tqdm(shuffled_samples[level][start_idx:], desc=f"Samples at Disorder Level {int(level*100)}%", leave=False), start=start_idx):
        print(f"Processing Sample {idx + 1}/{num_samples} at Disorder Level {int(level*100)}%")
        log_probs = get_predictability_scores(shuffled_text, sample_length - 1)
        if log_probs:
            results[level].append(log_probs)
        # Save progress every 10 samples
        if (idx + 1) % 10 == 0:
            completed_samples[level] = idx + 1
            with open('results.pkl', 'wb') as file:
                pickle.dump({'results': results, 'completed_samples': completed_samples}, file)
            print(f"Saved progress after processing {idx + 1} samples.")


# 2.5 Save final results

with open('results.pkl', 'wb') as file:
    pickle.dump({'results': results, 'completed_samples': completed_samples}, file)
print("Final results saved.")


# 2.6 Print results for verification

for level in disorder_levels:
    print(f"Disorder Level: {int(level*100)}%")
    for sample_idx, log_probs in enumerate(results[level]):
        print(f"  Sample {sample_idx + 1}: {log_probs}")
        if len(log_probs) != sample_length - 1:
            print(f"    Warning: Sample {sample_idx + 1} has {len(log_probs)} log probs, expected {sample_length - 1}")


# 2.7 Calculate average log probabilities

average_log_probs = {level: [] for level in disorder_levels}

for level in disorder_levels:
    context_length_sums = [0] * (sample_length - 1) # List to store the sum of log probs
    context_length_counts = [0] * (sample_length - 1) # list to store the count of log probs
    for sample_log_probs in results[level]:
        for i, log_prob in enumerate(sample_log_probs):
            context_length_sums[i] += log_prob # Add log prob to the sum
            context_length_counts[i] += 1 # Increase the count
    average_log_probs[level] = [context_length_sums[i] / context_length_counts[i] for i in range(sample_length - 1)]

# Create a matrix to store the average log probabilities (levels x context lengths)
matrix = np.array([average_log_probs[level] for level in disorder_levels]) 
#print(matrix)


# 2.8 Plot the log prob by disorder level and context length

plt.figure(figsize=(10, 8))
plt.imshow(matrix[::-1], cmap='viridis_r', aspect='auto')
plt.colorbar(label='Average Log Probability')
plt.xlabel('Context Length')
plt.ylabel('Disorder Level')
plt.title('Average Log Probability by Disorder Level and Context Length')
plt.xticks(ticks=np.arange(sample_length - 1), labels=np.arange(1, sample_length), rotation=90, fontsize=7)
plt.yticks(ticks=np.arange(len(disorder_levels)), labels=[f'{int(level*100)}%' for level in disorder_levels[::-1]])
file_path = '/Users/niyang/Downloads/analysis/logprob_heatmap.png'
plt.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()


''' 3. Calculate the Spearman correlation'''

# 3.1. Calculate Spearman correlation and p-value for each disorder level

spearman_results = []
for level in disorder_levels:
    log_probs = average_log_probs[level] 
    context_lengths = np.arange(1, sample_length)
    correlation, p_value = spearmanr(context_lengths, log_probs)
    spearman_results.append((level, correlation, p_value))

# Extract slopes to show how they change with disorder level
correlations = [result[1] for result in spearman_results] 
p_values = [result[2] for result in spearman_results]

# Make p-values easier to read
formatted_p_values = ["{:.2e}".format(p) if p < 0.01 else "{:.3f}".format(p) for p in p_values]

# Create a df to store the results
results_df = pd.DataFrame({
    'Disorder Level': [f'{int(level * 100)}%' for level in disorder_levels],
    'Spearman Correlation': correlations,
    'P-Value': formatted_p_values
})


# 3.2 Plotting the relationship between Spearman Correlation and disorder level

plt.figure(figsize=(10, 6))
sns.lineplot(x=[int(level * 100) for level in disorder_levels], y=correlations, marker='o', linestyle='-')
plt.xlabel('Disorder Level (%)')
plt.ylabel('Spearman Correlation')
plt.title('Relationship between Spearman Correlation and Disorder Level')
plt.grid(True)
plt.xticks(ticks=[int(level * 100) for level in disorder_levels])
plt.tight_layout()
file_path = '/Users/niyang/Downloads/analysis/spearman_linechart.png'
plt.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()


''' 4. Fit a regression model to check for ineteraction effect'''

# 4.1 Prepare data for the df

context_lengths = np.tile(np.arange(1, sample_length), len(disorder_levels))
disorder_levels_flat = np.repeat(disorder_levels, sample_length - 1)
average_log_probs_flat = matrix.flatten()
assert len(context_lengths) == len(disorder_levels_flat) == len(average_log_probs_flat)
data = {
    'context_length': context_lengths,
    'disorder_level': disorder_levels_flat,
    'average_log_prob': average_log_probs_flat
}
df = pd.DataFrame(data)


# 4.2 Fit the model with the interaction term

df['interaction'] = df['context_length'] * df['disorder_level']
X = df[['context_length', 'disorder_level', 'interaction']]
X = sm.add_constant(X)
y = df['average_log_prob']
model2 = sm.OLS(y, X).fit()
print(model2.summary())


# 4.3 Check residuals

residuals = model2.resid
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
file_path = '/Users/niyang/Downloads/analysis/histo_res.png'
plt.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()

sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
file_path = '/Users/niyang/Downloads/analysis/QQ_plot.png'
plt.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()


# 4.4 Plot the interaction effect

sns.lmplot(x='context_length', y='average_log_prob', hue='disorder_level', data=df, aspect=2, height=6, palette='viridis')
plt.title('Interaction Effect of Context Length and Disorder Level on Average Log Probability')
plt.xlabel('Context Length')
plt.ylabel('Average Log Probability')
file_path = '/Users/niyang/Downloads/analysis/Interaction Effect.png'
plt.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()







'''# Check for mediation effect(abandoned)
df['const'] = 1 # Add a constant term for intercept in regression models
# Run mediation analysis using Process from pyprocessmacro
model1 = Process(data=df, 
                y='average_log_prob', 
                x='context_length', 
                m=['disorder_level'], 
                model=4, 
                bootstrap=5000)
print(model1.summary())
# Path diagram for mediation analysis
plt.figure(figsize=(8, 6))
sns.regplot(x='context_length', y='average_log_prob', data=df, scatter_kws={'s': 20}, line_kws={'color': 'blue'})
plt.title('Direct Effect of Context Length on Average Log Probability')
plt.xlabel('Context Length')
plt.ylabel('Average Log Probability')
file_path = '/Users/niyang/Downloads/analysis/Direct Effect.png'
plt.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()

# Boxplot of disorder level vs. average log-prob
plt.figure(figsize=(8, 6))
sns.boxplot(x='disorder_level', y='average_log_prob', data=df)
plt.title('Disorder Level vs. Average Log Probability')
plt.xlabel('Disorder Level')
plt.ylabel('Average Log Probability')
file_path = '/Users/niyang/Downloads/analysis/boxplot.png'
plt.savefig(file_path, dpi=600, bbox_inches='tight')
plt.show()'''