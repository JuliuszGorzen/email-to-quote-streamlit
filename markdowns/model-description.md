**Name**

```text
gpt-3.5-turbo-0613
```

**Token Limit (prompt + response)**

```text
4096 tokens
-> 100 tokens ~= 75 words
-> 4096 / 100 = 40,96
-> 40,96 * 75 = 3072 words
```

**Dataset structure for fine-tuning (JSONL format):**

```json lines
{"messages": [{"role": "system", "content": "You are...."}, {"role": "user", "content": "Email example 1"}, {"role": "assistant", "content": "JSON response 1"}]}
{"messages": [{"role": "system", "content": "You are...."}, {"role": "user", "content": "Email example 2"}, {"role": "assistant", "content": "JSON response 2"}]}
{"messages": [{"role": "system", "content": "You are...."}, {"role": "user", "content": "Email example 3"}, {"role": "assistant", "content": "JSON response 3"}]}
.
.
.
```

**Prompt structure (Keywords)**

```text
System: -> Description of the bot
User: -> User input
Assistant: -> Bot response
```

**Documentation**

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [OpenAI text generation description](https://platform.openai.com/docs/guides/text-generation)
- [OpenAI prompt engineering](https://platform.openai.com/docs/guides/prompt-engineering)