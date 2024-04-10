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


def SR_FB_1_1(story_content, messages):
    prompt = f"""
    “{story_content}”
    为了确保文本的质量，我们邀请了两位评估员对上述文本进行全面评估：Mike和John。你是Mike，请根据以下五个维度评估文本：'重复问题'、'用词不当'、'逻辑问题'、'连贯性问题'和'事实性错误'。

    在我们的评估中，每出现一处错误扣分：-1分,起始分数:0分。
    其中“用词不当”包括不恰当的量词或者动词，以及句子中的语法和语义问题。“事实性错误”指故事中违反普遍接受的常识的描述。
    注意：每种错误类型在文本中可能出现多次,每出现一次扣分-1分,但“重复问题”（内容完全重复的实例）仅受到一次处罚，无论重复次数如何。
    
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


def SR_FB_1_2(story_content, context, reflection, messages):
    prompt = f"""
        “{story_content}”
        您好，John。上述文本需要您和另一位评估员Mike进行评估。您们都需要从以下几个维度深入评估文本：'重复问题'、'逻辑问题'、'连贯性问题'、'用词不当'和'事实性错误'。
        
        在我们的评估中，每出现一处错误扣分：-1分,起始分数:0分。
        其中“用词不当”包括不恰当的量词或者动词，以及句子中的语法和语义问题。“事实性错误”指故事中违反普遍接受的常识的描述。
        注意：每种错误类型在文本中可能出现多次,每出现一次扣分-1分,但“重复问题”（内容完全重复的实例）仅受到一次处罚，无论重复次数如何。
        
        以下是Mike对这些问题的看法：
        "{context}"
        "\n"
        "{reflection}"
        您有任何不同的意见吗？
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


def SR_FB_2_1(feedback, messages):
    prompt = f"""
        您好，Mike，在上一轮讨论中，一位评估员对于讨论过程提出下面的建议：
        "{feedback}"
        请根据他的建议在这一轮讨论中进行回应和改进。
        请继续进行您的分析。
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


def SR_FB_2_2(feedback, context, reflection, messages):
    prompt = f"""
        您好，John，在上一轮讨论中，一位评估员对于讨论过程提出下面的建议： 
        "{feedback}"
        请根据他的建议在这一轮讨论中进行回应和改进。
        Mike有如下的观点:
        "{context}"
        "\n"
        "{reflection}"
        你对此有什么意见吗? 
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


def SR_FB_3_1(feedback, messages):
    prompt = f"""
        您好，Mike，在上一轮讨论中，一位评估员对于讨论过程提出下面的建议：
        "{feedback}"
        请根据他的建议在这一轮讨论中进行回应和改进。
        请继续您的分析。
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


def SR_FB_3_2(feedback, context, reflection, messages):
    prompt = f"""
        您好，John，在上一轮讨论中，一位评估员对于讨论过程提出下面的建议： 
        "{feedback}"
        请根据他的建议在这一轮讨论中进行回应和改进。
        
        Mike对此有如下的观点:
        "{context}"
        "\n"
        "{reflection}"
        你有其他的想法吗John?
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


def Self_Reflection_key(messages):
    prompt = f"""
    针对之前讨论的故事文本，希望您仔细重新考虑并验证您所识别的问题是否确实存在。是否还有其他地方存在这种类型的问题？
    请注意，故事文本中可能在多个地方包含相同的错误，例如，文本的多个部分都可能存在“逻辑问题”。
    同时请您参考借鉴另一位评估者的发言，反思自己的评估。如果这类问题确实存在，请相应地修改您对叙述文本的评估。
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
    作为评估员July，请对这轮讨论进行全面评估。（字数必须控制在150个词内）

    在讨论中是否有遗漏或误解？是否有冗余或无关的对话？您建议下一轮讨论改进哪些方面？
    如果对某个问题存在分歧，引导他们在下一轮讨论中达成共识，尽量减少分歧。
    请提供全面的评估和具体建议，以指导我们下一次讨论的改进。
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


def SR_FB_summary(story_content, history, messages):
    prompt = f"""
            "{history}"

            请汇总上述提供两位评审员三轮的讨论，全面的总结评估以下的故事文本：
            
            “{story_content}”

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
        with open('LOT_SR+FB.json', 'r',
                  encoding='utf-8') as file:
            SR_FB_data = json.load(file)
    except FileNotFoundError:
        SR_FB_data = []

    try:
        with open('LOT_SR+FB.pkl',
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
        messages_agent2 = [{"role": "system", "content": ROLE}]
        messages_agent3 = [{"role": "system", "content": ROLE}]
        messages_agent4 = [{"role": "system", "content": ROLE}]

        while True:
            try:
                # 第一轮
                # 第一个人
                SR_FB_agent1_1 = SR_FB_1_1(story["story"], messages_agent1)
                messages_agent1.append({"role": "user", "content": SR_FB_agent1_1})

                Self_Reflection_agent1_1 = Self_Reflection_key(messages_agent1)
                messages_agent1.append(
                    {"role": "user", "content": Self_Reflection_agent1_1})

                # 第二个人
                SR_FB_agent1_2 = SR_FB_1_2(story["story"], SR_FB_agent1_1, Self_Reflection_agent1_1,
                                                   messages_agent2)
                messages_agent2.append({"role": "user",
                                        "content": SR_FB_agent1_1 + "\n" + " " + Self_Reflection_agent1_1 + "\n" + " " + SR_FB_agent1_2})

                Self_Reflection_agent1_2 = Self_Reflection_key(messages_agent2)
                messages_agent2.append(
                    {"role": "user", "content": Self_Reflection_agent1_2})

                messages_agent1.append({"role": "user", "content": SR_FB_agent1_2 + "\n" + " " + Self_Reflection_agent1_2})

                # 反馈
                history1 = SR_FB_agent1_1 + "\n" + " " + Self_Reflection_agent1_1 + "\n" + " " + SR_FB_agent1_2 + "\n" + " " + Self_Reflection_agent1_2
                feedback_1 = feed_back(history1, messages_agent4)
                messages_agent4.append({"role": "user", "content": feedback_1})

                # 第二轮
                # 第一人
                SR_FB_agent2_1 = SR_FB_2_1(feedback_1, messages_agent1)
                messages_agent1.append(
                    {"role": "user", "content": SR_FB_agent2_1})

                # 反思
                Self_Reflection_agent2_1 = Self_Reflection_key(messages_agent1)
                messages_agent1.append(
                    {"role": "user", "content": Self_Reflection_agent2_1})

                # 第二个人
                SR_FB_agent2_2 = SR_FB_2_2(feedback_1, SR_FB_agent2_1, Self_Reflection_agent2_1,
                                                   messages_agent2)
                messages_agent2.append({"role": "user",
                                        "content": SR_FB_agent2_1 + "\n" + " " + Self_Reflection_agent2_1 + "\n" + " " + SR_FB_agent2_2})

                # 反思
                Self_Reflection_agent2_2 = Self_Reflection_key(messages_agent2)
                messages_agent2.append(
                    {"role": "user", "content": Self_Reflection_agent2_2})

                messages_agent1.append(
                    {"role": "user", "content": SR_FB_agent2_2 + "\n" + " " + Self_Reflection_agent2_2})
                # 反馈
                history2 = SR_FB_agent2_1 + "\n" + " " + Self_Reflection_agent2_1 + "\n" + " " + SR_FB_agent2_2 + "\n" + " " + Self_Reflection_agent2_2
                feedback_2 = feed_back(history2, messages_agent4)
                messages_agent4.append({"role": "user", "content": feedback_2})

                # 第三轮
                # 第一人
                SR_FB_agent3_1 = SR_FB_3_1(feedback_2, messages_agent1)
                messages_agent1.append(
                    {"role": "user", "content": SR_FB_agent3_1})

                Self_Reflection_agent3_1 = Self_Reflection_key(messages_agent1)
                messages_agent1.append(
                    {"role": "user", "content": Self_Reflection_agent3_1})

                # 第二个人
                SR_FB_agent3_2 = SR_FB_3_2(feedback_2, SR_FB_agent3_1, Self_Reflection_agent3_1,
                                                   messages_agent2)

                messages_agent2.append(
                    {"role": "user",
                     "content": SR_FB_agent3_1 + "\n" + " " + Self_Reflection_agent3_1 + "\n" + " " + SR_FB_agent3_2})

                Self_Reflection_agent3_2 = Self_Reflection_key(messages_agent2)

                messages_agent2.append(
                    {"role": "user", "content": Self_Reflection_agent3_2})

                # 总结
                his = "Mike：" + Self_Reflection_agent1_1 + "\n " + "John：" + Self_Reflection_agent1_2 + "\n " + "Mike：" + Self_Reflection_agent2_1 + "\n " + "John：" + Self_Reflection_agent2_2 + "\n " + "Mike：" + Self_Reflection_agent3_1 + "\n " + "John：" + Self_Reflection_agent3_2 + "\n "
                SR_FB_agent_summary = SR_FB_summary(story["story"], his, messages_agent3)

                summary = text_summary(SR_FB_agent_summary)

                SR_FB_data.append({
                    "ID": idx,
                    "prompt": story["prompt"],
                    "story": story["story"],
                    "SR_FB_agent1_1": SR_FB_agent1_1,
                    "SR_FB_Reflection1_1": Self_Reflection_agent1_1,
                    "SR_FB_agent1_2": SR_FB_agent1_2,
                    "SR_FB_Reflection1_2": Self_Reflection_agent1_2,
                    "Feedback_1": feedback_1,
                    "SR_FB_agent2_1": SR_FB_agent2_1,
                    "SR_FB_Reflection2_1": Self_Reflection_agent2_1,
                    "SR_FB_agent2_2": SR_FB_agent2_2,
                    "SR_FB_Reflection2_2": Self_Reflection_agent2_2,
                    "Feedback_2": feedback_2,
                    "SR_FB_agent3_1": SR_FB_agent3_1,
                    "SR_FB_Reflection3_1": Self_Reflection_agent3_1,
                    "SR_FB_agent3_2": SR_FB_agent3_2,
                    "SR_FB_Reflection3_2": Self_Reflection_agent3_2,
                    "SR_FB_agent_summary": SR_FB_agent_summary,
                    "text_summary": summary
                })

                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {idx + 1}/{len(stories)} prompts")
                print(SR_FB_data[-1])

                with open('LOT_SR+FB.json', 'w',
                          encoding='utf-8') as file:
                    json.dump(SR_FB_data, file, ensure_ascii=False, indent=4)

                with open('LOT_SR+FB.pkl',
                          'wb') as progress_file:
                    pickle.dump(idx + 1, progress_file)

                break
            except (UnboundLocalError, KeyError) as e:
                print(f"Error occurred: {e}. Retrying...")
                time.sleep(20)

    print("Processing completed and saved!")


if __name__ == '__main__':
    main()
