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

error_types = ['Repetition', 'Inappropriate Lexical Choice', 'Logical Inconsistency', 'Discontinuity', 'Factual Error']


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


def one_by_one_1_1(story_content, messages):
    prompt = f"""
    “{story_content}”
    To ensure the quality of the text, we have invited two evaluators to conduct a comprehensive assessment of the aforementioned text: Mike and John. and you are Mike. Please evaluate the text based on the following five dimensions: 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'.

    In our evaluation,  each error results in a deduction of one point, starting from zero. 
    "Inappropriate Lexical Choice" includes the improper use of quantifiers or verbs, as well as grammatical and semantic issues within sentences. 
    "Factual Errors" refer to descriptions in the story that contradict universally accepted common knowledge. 
    Note: While each type of error may occur multiple times within the text, leading to a deduction of one point per occurrence, 'Repetition Issues' - instances of exact duplication in content - are only penalized once, regardless of the number of repetitions.
    """
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


def one_by_one_1_2(story_content, context, messages):
    prompt = f"""
        “{story_content}”
        Hi John. The aforementioned text awaits evaluation by both you and another evaluator, Mike. You both are required to deeply assess the text from the following dimensions: 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'. 

        In our evaluation,  each error results in a deduction of one point, starting from zero. 
        "Inappropriate Lexical Choice" includes the improper use of quantifiers or verbs, as well as grammatical and semantic issues within sentences. 
        "Factual Errors" refer to descriptions in the story that contradict universally accepted common knowledge. 
        Note: While each type of error may occur multiple times within the text, leading to a deduction of one point per occurrence, 'Repetition Issues' - instances of exact duplication in content - are only penalized once, regardless of the number of repetitions.

        Here are Mike's views on these issues:
        "{context}"
        "\n"
        Do you have any differing opinions concerning these dimensions?
        """
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


def one_by_one_2_1(feedback, messages):
    prompt = f"""
        Hello Mike, During the last round of discussions, an evaluator provided the following suggestions for discussion: 
        "{feedback}"
        Take his advice and address it in this round of discussion. 
        Please proceed with your analysis.
        """
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


def one_by_one_2_2(feedback, context, messages):
    prompt = f"""
        Hi John, 
        During the last round of discussions, an evaluator provided the following suggestions for discussion: 
        "{feedback}"
        Take his advice and address it in this round of discussion.
        Mike has the following opinion:
        "{context}"
        "\n"
        What are your thoughts on this? 
        """
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


def one_by_one_3_1(feedback, messages):
    prompt = f"""
        Hello Mike, During the last round of discussions, an evaluator provided the following suggestions for discussion: 
        "{feedback}"
        Take his advice and address it in this round of discussion.
        Please continue your analysis.
        """
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


def one_by_one_3_2(feedback, context, messages):
    prompt = f"""
        Hello John, 
        During the last round of discussions, an evaluator provided the following suggestions for discussion: 
        "{feedback}"
        Take his advice and address it in this round of discussion.
        Mike has the following opinion:
        "{context}"
        "\n"
        What are your thoughts on this, John?
        """
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


def feed_back(history, messages):
    prompt = f"""
    "{history}"
    As evaluator July, please conduct a comprehensive assessment of this round of discussion. (The word count must be controlled at 150 tokens)

    Were there any omissions or misunderstandings? Were there any redundant or irrelevant dialogues? What areas do you suggest improving for the next round?
    If there are disagreements on a certain issue, guide them to reach a consensus in the next round of discussion, aiming to minimize differences.
    Please provide a holistic evaluation and specific recommendations to guide improvements in our next discussion.
    """
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


def one_by_one_summary(story_content, history, messages):
    prompt = f"""
            "{history}"

            Based on the provided story text and the evaluations from the two previous reviewers across three rounds, please provide a comprehensive assessment.

            “{story_content}”

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

    max_tokens_messages = 2200
    max_tokens_prompt = 2200
    encoding_name = "cl100k_base"
    if num_tokens_from_string(prompt, encoding_name) > max_tokens_prompt:
        prompt = truncate_text_from_start(prompt, max_tokens_prompt, encoding_name)
    d = {"role": "user", "content": prompt}
    messages.append(d)
    retries = 20
    max_tokens_result = 1000
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
        except (openai.InternalServerError, UnboundLocalError, KeyError) as e:
            print(f"Error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(40)

    raise Exception("Failed after multiple retries.")

def main():
    try:
        with open('WP_one_by_one+FB.json', 'r',
                  encoding='utf-8') as file:
            one_by_one_data = json.load(file)
    except FileNotFoundError:
        one_by_one_data = []

    try:
        with open('WP_one_by_one+FB.pkl',
                  'rb') as progress_file:
            start_index = pickle.load(progress_file)
    except FileNotFoundError:
        start_index = 0

    with open('WP_test_change_final.json', 'r',
              encoding='utf-8') as file:
        stories = json.load(file)

    for idx, story in enumerate(stories[start_index:]):
        idx += start_index

        messages_agent1 = [{"role": "system", "content": ROLE}]
        messages_agent2 = [{"role": "system", "content": ROLE}]
        messages_agent3 = [{"role": "system", "content": ROLE}]
        messages_agent4 = [{"role": "system", "content": ROLE}]

        while True:
            try:
                # 第一轮
                # 第一个人
                one_by_one_agent1_1 = one_by_one_1_1(story["story"], messages_agent1)
                messages_agent1.append({"role": "user", "content": one_by_one_agent1_1})

                # 第二个人
                one_by_one_agent1_2 = one_by_one_1_2(story["story"], one_by_one_agent1_1,
                                                   messages_agent2)
                messages_agent2.append({"role": "user",
                                        "content": one_by_one_agent1_1 + "\n" + " " + one_by_one_agent1_2})

                messages_agent1.append({"role": "user", "content": one_by_one_agent1_2})

                # 反馈
                history1 = one_by_one_agent1_1 + "\n" + " " + one_by_one_agent1_2
                feedback_1 = feed_back(history1, messages_agent4)
                messages_agent4.append({"role": "user", "content": feedback_1})

                # 第二轮
                # 第一人
                one_by_one_agent2_1 = one_by_one_2_1(feedback_1, messages_agent1)
                messages_agent1.append(
                    {"role": "user", "content": one_by_one_agent2_1})

                # 第二个人
                one_by_one_agent2_2 = one_by_one_2_2(feedback_1, one_by_one_agent2_1,
                                                   messages_agent2)
                messages_agent2.append({"role": "user",
                                        "content": one_by_one_agent2_1 + "\n" + " " + one_by_one_agent2_2})

                messages_agent1.append({"role": "user", "content": one_by_one_agent2_2})

                # 反馈
                history2 = one_by_one_agent2_1 + "\n" + " " + one_by_one_agent2_2
                feedback_2 = feed_back(history2, messages_agent4)
                messages_agent4.append({"role": "user", "content": feedback_2})

                # 第三轮
                # 第一人
                one_by_one_agent3_1 = one_by_one_3_1(feedback_2, messages_agent1)
                messages_agent1.append(
                    {"role": "user", "content": one_by_one_agent3_1})

                # 第二个人
                one_by_one_agent3_2 = one_by_one_3_2(feedback_2, one_by_one_agent3_1,
                                                   messages_agent2)
                messages_agent2.append(
                    {"role": "user",
                     "content": one_by_one_agent3_1 + "\n" + " " + one_by_one_agent3_2})

                # 总结
                his = "Mike：" + one_by_one_agent1_1 + "\n " + "John：" + one_by_one_agent1_2 + "\n " + "Mike：" + one_by_one_agent2_1 + "\n " + "John：" + one_by_one_agent2_2 + "\n " + "Mike：" + one_by_one_agent3_1 + "\n " + "John：" + one_by_one_agent3_2 + "\n "
                one_by_one_agent_summary = one_by_one_summary(story["story"], his, messages_agent3)
                summary = text_summary(one_by_one_agent_summary)

                one_by_one_data.append({
                    "ID": idx,
                    "prompt": story["prompt"],
                    "story": story["story"],
                    "one_by_one_agent1_1": one_by_one_agent1_1,
                    "one_by_one_agent1_2": one_by_one_agent1_2,
                    "Feedback_1": feedback_1,
                    "one_by_one_agent2_1": one_by_one_agent2_1,
                    "one_by_one_agent2_2": one_by_one_agent2_2,
                    "Feedback_2": feedback_2,
                    "one_by_one_agent3_1": one_by_one_agent3_1,
                    "one_by_one_agent3_2": one_by_one_agent3_2,
                    "one_by_one_agent_summary": one_by_one_agent_summary,
                    "text_summary": summary
                })

                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {idx + 1}/{len(stories)} prompts")
                print(one_by_one_data[-1])

                with open('WP_one_by_one+FB.json', 'w',
                          encoding='utf-8') as file:
                    json.dump(one_by_one_data, file, ensure_ascii=False, indent=4)

                with open('WP_one_by_one+FB.pkl',
                          'wb') as progress_file:
                    pickle.dump(idx + 1, progress_file)

                break
            except (UnboundLocalError, KeyError) as e:
                print(f"Error occurred: {e}. Retrying...")
                time.sleep(20)

    print("Processing completed and saved!")


if __name__ == '__main__':
    main()
