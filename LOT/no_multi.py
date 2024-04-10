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
    请把以下的评估QA对总结为一段完整通顺的文本评估段落，言简意赅：
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
            
            为了确保文本的质量，请根据以下五个维度评估文本：'重复问题'、'用词不当'、'逻辑问题'、'连贯性问题'和'事实性错误'。
            注意：文本中每种错误类型可能出现多次。每个实例都应被识别并获得相应得分。
            我们将从故事的基础分0分开始。对于每个检测到的错误扣除-1分。
    
            请只需回答以下布尔类型问题（是/否）严格按照以下格式：如果答案是“是”，列出具体的原句，解释问题，并根据错误类型和严重性说明扣分。如果有多处存在该问题，请全部列出。如果答案是“否”，则无需进一步解释。“单项问题得分”得分指的是特指这个问题的扣分。
    
            问：上述故事中是否包含'重复问题'？
            答：
            单项问题得分：
        
            问：上述故事中是否包含'逻辑问题'？
            答：
            单项问题得分：
        
            问：上述故事中是否包含'连贯性问题'？
            答：
            单项问题得分：
        
            问：上述故事中是否包含'用词不当'？
            答：
            单项问题得分：
        
            问：上述故事中是否包含'事实性错误'？
            答：
            单项问题得分：
        
            最终得分：_ 分
            计算过程：
        

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

    with open('LOT_test_change_final.json', 'r',
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
