import json
import time
import pickle
from datetime import datetime
import testService
import tiktoken

class TextEvaluator:
    def __init__(self):
        self.ROLE = 'assistant'
        self.USER_ROLE = 'user'
        self.error_types = ['Repetition', 'Inappropriate Lexical Choice', 'Logical Inconsistency', 'Discontinuity', 'Factual Error']
        self.retries = 20
        self.max_tokens_messages = 2500
        self.max_tokens_result = 600
        self.encoding_name = "cl100k_base"

    def num_tokens_from_string(self, string):
        encoding = tiktoken.get_encoding(self.encoding_name)
        return len(encoding.encode(string))

    def truncate_text_from_start(self, text, max_tokens):
        tokens = tiktoken.get_encoding(self.encoding_name).encode(text)
        while len(tokens) > max_tokens:
            tokens.pop(0)
        return tiktoken.get_encoding(self.encoding_name).decode(tokens)

    def truncate_messages(self, messages, max_tokens):
        total_tokens = sum(self.num_tokens_from_string(message['content']) for message in messages)
        while total_tokens > max_tokens:
            first_message = messages.pop(0)
            total_tokens -= self.num_tokens_from_string(first_message['content'])
        return messages

    def replace_special_chars(self, text):
        replacements = {
            "&lsquo;": "‘", "&rsquo;": "’",
            "&ldquo;": "“", "&rdquo;": "”"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def retry_request(self, messages):
        for _ in range(self.retries):
            try:
                result = testService.request_content(messages)
                result = self.replace_special_chars(result)
                if self.num_tokens_from_string(result) > self.max_tokens_result:
                    result = self.truncate_text_from_start(result, self.max_tokens_result)
                return result
            except (UnboundLocalError, KeyError) as e:
                print(f"Error occurred: {e}. Retrying...")
                time.sleep(20)
        raise Exception("Failed after multiple retries.")

    def SR_COT_FB_1_1(self, story_content, messages):
        prompt = f"""
            “{story_content}”
            To ensure the quality of the text, we have invited two evaluators to conduct a comprehensive assessment of the aforementioned text: Mike and John. and you are Mike. Please evaluate the text based on the following five dimensions: 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'.

            In our evaluation,  each error results in a deduction of one point, starting from zero. 
            "Inappropriate Lexical Choice" includes the improper use of quantifiers or verbs, as well as grammatical and semantic issues within sentences. 
            "Factual Errors" refer to descriptions in the story that contradict universally accepted common knowledge. 
            Note: While each type of error may occur multiple times within the text, leading to a deduction of one point per occurrence, 'Repetition Issues' - instances of exact duplication in content - are only penalized once, regardless of the number of repetitions.
            Now, let's think step by step.
            What do you believe are the two most discussable issues in the aforementioned text? Please first assess the text focusing on these two issues.

            """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def SR_COT_FB_1_2(self, story_content, context, reflection, messages):
        prompt = f"""
                “{story_content}”
                Hi John. The aforementioned text awaits evaluation by both you and another evaluator, Mike. You both are required to deeply assess the text from the following dimensions: 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'. 

                In our evaluation,  each error results in a deduction of one point, starting from zero. 
                "Inappropriate Lexical Choice" includes the improper use of quantifiers or verbs, as well as grammatical and semantic issues within sentences. 
                "Factual Errors" refer to descriptions in the story that contradict universally accepted common knowledge. 
                Note: While each type of error may occur multiple times within the text, leading to a deduction of one point per occurrence, 'Repetition Issues' - instances of exact duplication in content - are only penalized once, regardless of the number of repetitions.

                Currently, we will only discuss the two most prominent issues. Here are Mike's views on these two issues:
                "{context}"
                "\n"
                "{reflection}"
                Do you have any differing opinions concerning these dimensions?
                """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def SR_COT_FB_2_1(self, feedback, messages):
        prompt = f"""
            Hello Mike, During the last round of discussions, an evaluator provided the following suggestions for discussion: 
            "{feedback}"
            Take his advice and address it in this round of discussion. 
            Next, in addition to the two main issues previously considered, we need to contemplate two additional aspects from the following: 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'.
            Please proceed with your analysis.
            """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def SR_COT_FB_2_2(self, feedback, context, reflection, messages):
        prompt = f"""
            Hi John, 
            During the last round of discussions, an evaluator provided the following suggestions for discussion: 
            "{feedback}"
            Take his advice and address it in this round of discussion.
            Next, in addition to the two main issues previously considered, we need to contemplate two additional aspects from the following: 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'.
            Mike has the following opinion:
            "{context}"
            "\n"
            "{reflection}"
            What are your thoughts on this? 
            """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def SR_COT_FB_3_1(self, feedback, messages):
        prompt = f"""
            Hello Mike, During the last round of discussions, an evaluator provided the following suggestions for discussion: 
            "{feedback}"
            Take his advice and address it in this round of discussion.

            Next, we need to continue discussing the remaining last issue among 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error'.
            Please continue your analysis.
            """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def SR_COT_FB_3_2(self, feedback, context, reflection, messages):
        prompt = f"""
            Hello John, 
            During the last round of discussions, an evaluator provided the following suggestions for discussion: 
            "{feedback}"
            Take his advice and address it in this round of discussion.
            Next, we need to continue discussing the remaining last issue among 'Repetition', 'Logical Inconsistency', 'Discontinuity', 'Inappropriate Lexical Choice', and 'Factual Error', 
            Mike has the following opinion:
            "{context}"
            "\n"
            "{reflection}"
            What are your thoughts on this, John?
            """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    # 同样地，为其他 SR_COT_FB_x_x 方法实现逻辑
    # ...

    def Self_Reflection_key(self, messages):
        prompt = f"""
        Regarding the narrative text previously discussed, I would like you to carefully reconsider and verify whether the issues of the type you identified indeed exist.
        """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def feed_back(self, history, messages):
        prompt = f"""
        "{history}"
        As evaluator July, please conduct a comprehensive assessment of this round of discussion.
        """
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def text_summary(self, mes):
        prompt = f"""
        Please summarize the following assessment Q&A pairs into a coherent and concise evaluative paragraph:
        {mes}
        """
        d = {"role": "user", "content": prompt}
        messages = [{"role": "system", "content": self.ROLE}, d]
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def SR_COT_FB_summary(self, story_content, history, messages):
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

        # ... 构建信息和调用重试请求的逻辑 ...
        d = {"role": "user", "content": prompt}
        messages.append(d)
        messages = self.truncate_messages(messages, self.max_tokens_messages)
        return self.retry_request(messages)

    def main(self):
        try:
            with open('ROC_SR+COT+FB.json', 'r',
                      encoding='utf-8') as file:
                SR_COT_FB_data = json.load(file)
        except FileNotFoundError:
            SR_COT_FB_data = []

        try:
            with open('ROC_SR+COT+FB.pkl',
                      'rb') as progress_file:
                start_index = pickle.load(progress_file)
        except FileNotFoundError:
            start_index = 0

        with open('ROC_test_change_final.json', 'r',
                  encoding='utf-8') as file:
            stories = json.load(file)

        for idx, story in enumerate(stories[start_index:]):
            idx += start_index

            messages_agent1 = [{"role": "system", "content": self.ROLE}]
            messages_agent2 = [{"role": "system", "content": self.ROLE}]
            messages_agent3 = [{"role": "system", "content": self.ROLE}]
            messages_agent4 = [{"role": "system", "content": self.ROLE}]

            while True:
                try:
                    # 第一轮
                    # 第一个人
                    SR_COT_FB_agent1_1 = self.SR_COT_FB_1_1(story["story"], messages_agent1)
                    messages_agent1.append({"role": "user", "content": SR_COT_FB_agent1_1})

                    Self_Reflection_agent1_1 = self.Self_Reflection_key(messages_agent1)
                    messages_agent1.append(
                        {"role": "user", "content": Self_Reflection_agent1_1})

                    # 第二个人
                    SR_COT_FB_agent1_2 = self.SR_COT_FB_1_2(story["story"], SR_COT_FB_agent1_1, Self_Reflection_agent1_1,
                                                       messages_agent2)
                    messages_agent2.append({"role": "user",
                                            "content": SR_COT_FB_agent1_1 + "\n" + " " + Self_Reflection_agent1_1 + "\n" + " " + SR_COT_FB_agent1_2})

                    Self_Reflection_agent1_2 = self.Self_Reflection_key(messages_agent2)
                    messages_agent2.append(
                        {"role": "user", "content": Self_Reflection_agent1_2})

                    messages_agent1.append(
                        {"role": "user", "content": SR_COT_FB_agent1_2 + "\n" + " " + Self_Reflection_agent1_2})

                    # 反馈
                    history1 = SR_COT_FB_agent1_1 + "\n" + " " + Self_Reflection_agent1_1 + "\n" + " " + SR_COT_FB_agent1_2 + "\n" + " " + Self_Reflection_agent1_2
                    feedback_1 = self.feed_back(history1, messages_agent4)
                    messages_agent4.append({"role": "user", "content": feedback_1})

                    # 第二轮
                    # 第一人
                    SR_COT_FB_agent2_1 = self.SR_COT_FB_2_1(feedback_1, messages_agent1)
                    messages_agent1.append(
                        {"role": "user", "content": SR_COT_FB_agent2_1})

                    # 反思
                    Self_Reflection_agent2_1 = self.Self_Reflection_key(messages_agent1)
                    messages_agent1.append(
                        {"role": "user", "content": Self_Reflection_agent2_1})

                    # 第二个人
                    SR_COT_FB_agent2_2 = self.SR_COT_FB_2_2(feedback_1, SR_COT_FB_agent2_1, Self_Reflection_agent2_1,
                                                       messages_agent2)
                    messages_agent2.append({"role": "user",
                                            "content": SR_COT_FB_agent2_1 + "\n" + " " + Self_Reflection_agent2_1 + "\n" + " " + SR_COT_FB_agent2_2})

                    # 反思
                    Self_Reflection_agent2_2 = self.Self_Reflection_key(messages_agent2)
                    messages_agent2.append(
                        {"role": "user", "content": Self_Reflection_agent2_2})
                    messages_agent1.append(
                        {"role": "user", "content": SR_COT_FB_agent2_2 + "\n" + " " + Self_Reflection_agent2_2})
                    # 反馈
                    history2 = SR_COT_FB_agent2_1 + "\n" + " " + Self_Reflection_agent2_1 + "\n" + " " + SR_COT_FB_agent2_2 + "\n" + " " + Self_Reflection_agent2_2
                    feedback_2 = self.feed_back(history2, messages_agent4)
                    messages_agent4.append({"role": "user", "content": feedback_2})

                    # 第三轮
                    # 第一人
                    SR_COT_FB_agent3_1 = self.SR_COT_FB_3_1(feedback_2, messages_agent1)
                    messages_agent1.append(
                        {"role": "user", "content": SR_COT_FB_agent3_1})

                    Self_Reflection_agent3_1 = self.Self_Reflection_key(messages_agent1)
                    messages_agent1.append(
                        {"role": "user", "content": Self_Reflection_agent3_1})

                    # 第二个人
                    SR_COT_FB_agent3_2 = self.SR_COT_FB_3_2(feedback_2, SR_COT_FB_agent3_1, Self_Reflection_agent3_1,
                                                       messages_agent2)

                    messages_agent2.append(
                        {"role": "user",
                         "content": SR_COT_FB_agent3_1 + "\n" + " " + Self_Reflection_agent3_1 + "\n" + " " + SR_COT_FB_agent3_2})

                    Self_Reflection_agent3_2 = self.Self_Reflection_key(messages_agent2)

                    messages_agent2.append(
                        {"role": "user", "content": Self_Reflection_agent3_2})

                    # 总结
                    his = "Mike：" + Self_Reflection_agent1_1 + "\n " + "John：" + Self_Reflection_agent1_2 + "\n " + "Mike：" + Self_Reflection_agent2_1 + "\n " + "John：" + Self_Reflection_agent2_2 + "\n " + "Mike：" + Self_Reflection_agent3_1 + "\n " + "John：" + Self_Reflection_agent3_2 + "\n "
                    SR_COT_FB_agent_summary = self.SR_COT_FB_summary(story["story"], his, messages_agent3)

                    summary = self.text_summary(SR_COT_FB_agent_summary)

                    SR_COT_FB_data.append({
                        "ID": idx,
                        "prompt": story["prompt"],
                        "story": story["story"],
                        "SR_COT_FB_agent1_1": SR_COT_FB_agent1_1,
                        "SR_COT_FB_Reflection1_1": Self_Reflection_agent1_1,
                        "SR_COT_FB_agent1_2": SR_COT_FB_agent1_2,
                        "SR_COT_FB_Reflection1_2": Self_Reflection_agent1_2,
                        "Feedback_1": feedback_1,
                        "SR_COT_FB_agent2_1": SR_COT_FB_agent2_1,
                        "SR_COT_FB_Reflection2_1": Self_Reflection_agent2_1,
                        "SR_COT_FB_agent2_2": SR_COT_FB_agent2_2,
                        "SR_COT_FB_Reflection2_2": Self_Reflection_agent2_2,
                        "Feedback_2": feedback_2,
                        "SR_COT_FB_agent3_1": SR_COT_FB_agent3_1,
                        "SR_COT_FB_Reflection3_1": Self_Reflection_agent3_1,
                        "SR_COT_FB_agent3_2": SR_COT_FB_agent3_2,
                        "SR_COT_FB_Reflection3_2": Self_Reflection_agent3_2,
                        "SR_COT_FB_agent_summary": SR_COT_FB_agent_summary,
                        "text_summary": summary
                    })

                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processed {idx + 1}/{len(stories)} prompts")
                    print(SR_COT_FB_data[-1])

                    with open('ROC_SR+COT+FB.json', 'w',
                              encoding='utf-8') as file:
                        json.dump(SR_COT_FB_data, file, ensure_ascii=False, indent=4)

                    with open('ROC_SR+COT+FB.pkl',
                              'wb') as progress_file:
                        pickle.dump(idx + 1, progress_file)

                    break
                except (UnboundLocalError, KeyError) as e:
                    print(f"Error occurred: {e}. Retrying...")
                    time.sleep(20)

        print("Processing completed and saved!")


# 使用类
if __name__ == '__main__':
    evaluator = TextEvaluator()
    evaluator.main()