import openai
import json
import time
import pickle
from datetime import datetime
import testService
import tiktoken

ROLE = 'assistant'
USER_ROLE = 'user'
PROMPT = f'you are {ROLE}.'


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_text_from_start(text, max_tokens, encoding_name):
    tokens = tiktoken.get_encoding(encoding_name).encode(text)
    while len(tokens) > max_tokens:
        # Remove tokens from the start
        tokens.pop(0)
    return tiktoken.get_encoding(encoding_name).decode(tokens)


def truncate_messages(messages, max_tokens, encoding_name):
    total_tokens = 0
    for message in messages:
        total_tokens += num_tokens_from_string(message['content'], encoding_name)

    while total_tokens > max_tokens:
        first_message = messages.pop(0)
        total_tokens -= num_tokens_from_string(first_message['content'], encoding_name)

    return messages


def text_summary(mes):
    prompt = f"""
    Please summarize the following assessment Q&A pairs into a coherent and concise evaluative paragraph:
    "\n"
    {mes}
    """
    ROLE = 'assistant'
    messages = [{"role": "system", "content": ROLE}]
    d = {"role": "user", "content": prompt}
    messages.append(d)
    retries = 20
    max_tokens_messages = 2500
    max_tokens_result = 600
    encoding_name = "cl100k_base"
    messages = truncate_messages(messages, max_tokens_messages, encoding_name)
    while retries > 0:
        try:
            result = testService.request_content(messages)
            result = result.replace("&lsquo;", "‘").replace("&rsquo;", "’")
            result = result.replace("&ldquo;", "“").replace("&rdquo;", "”")

            # Check if result exceeds token limit and truncate if necessary
            if num_tokens_from_string(result, encoding_name) > max_tokens_result:
                result = truncate_text_from_start(result, max_tokens_result, encoding_name)

            return result
        except (UnboundLocalError, KeyError) as e:
            print(f"Error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(20)

    raise Exception("Failed after multiple retries.")


def no_multi(story_content, messages):
    prompt = f"""
            “{story_content}”
            To ensure the quality of the text, we have invited you to conduct a comprehensive assessment of the aforementioned text.  Please evaluate the text based on the following five dimensions: 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'.
            Note: Each error type may occur multiple times in the text. Each instance should be identified and scored accordingly.
            We will start with a base score of 0 for the story. For each detected error, 1 point will be deducted:

            You only need to answer the following Boolean type questions (Yes/No), and strictly follow this format: If the answer is "Yes," list the specific original sentences, explain the issue, and detail the deductions based on the type and severity of the error. If the issue occurs in multiple places, please list them all. If the answer is "No," no further explanation is required. "Score for individual questions" refers to the deduction for this question specifically.

            Question: Does the above story contain 'Repetition'?
            Answer:
            Score for individual questions:

            Question: Does the above story contain 'Logical Inconsistency'?
            Answer:
            Score for individual questions:

            Question: Does the above story contain 'Discontinuity'?
            Answer:
            Score for individual questions:

            Question: Does the above story contain 'Inappropriate Lexical Choice'?
            Answer:
            Score for individual questions:

            Question: Does the above story contain 'Factual Error'?
            Answer:
            Score for individual questions:

            Final Score: _ points
            Calculation process:

    """
    d = {"role": "user", "content": prompt}
    messages.append(d)
    retries = 20
    max_tokens_messages = 2500
    max_tokens_result = 1000
    encoding_name = "cl100k_base"
    messages = truncate_messages(messages, max_tokens_messages, encoding_name)
    while retries > 0:
        try:
            result = testService.request_content(messages)
            result = result.replace("&lsquo;", "‘").replace("&rsquo;", "’")
            result = result.replace("&ldquo;", "“").replace("&rdquo;", "”")

            # Check if result exceeds token limit and truncate if necessary
            if num_tokens_from_string(result, encoding_name) > max_tokens_result:
                result = truncate_text_from_start(result, max_tokens_result, encoding_name)

            return result
        except (UnboundLocalError, KeyError) as e:
            print(f"Error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(20)

    raise Exception("Failed after multiple retries.")

def main():
    try:
        with open('no_multi.json', 'r',
                  encoding='utf-8') as file:
            no_multi_data = json.load(file)
    except FileNotFoundError:
        no_multi_data = []

    try:
        with open('no_multi.pkl',
                  'rb') as progress_file:
            start_index = pickle.load(progress_file)
    except FileNotFoundError:
        start_index = 0

    with open('ROC_test_change_final.json', 'r',
              encoding='utf-8') as file:
        stories = json.load(file)

    for idx, story in enumerate(stories[start_index:]):
        idx += start_index

        messages_agent1 = [{"role": "system", "content": ROLE}]

        while True:
            try:
                no_multi_agent = no_multi(story["story"], messages_agent1)
                messages_agent1.append({"role": "user", "content": no_multi_agent})

                summary = text_summary(no_multi_agent)

                no_multi_data.append({
                    "ID": idx,
                    "prompt": story["prompt"],
                    "story": story["story"],
                    "no_multi_agent": no_multi_agent,
                    "text_summary": summary
                })

                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {idx + 1}/{len(stories)} prompts")
                print(no_multi_data[-1])

                with open('no_multi.json', 'w',
                          encoding='utf-8') as file:
                    json.dump(no_multi_data, file, ensure_ascii=False, indent=4)

                with open('no_multi.pkl',
                          'wb') as progress_file:
                    pickle.dump(idx + 1, progress_file)
                break
            except (UnboundLocalError, KeyError) as e:
                print(f"Error occurred: {e}. Retrying...")
                time.sleep(20)

    print("Processing completed and saved!")


if __name__ == '__main__':
    main()
