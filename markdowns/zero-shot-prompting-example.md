### *System message*

Message for priming AI behavior, usually passed in as the first of a sequence of input messages.

```text
You are a bot that extract data from emails in different languages. Below are the rules that you have to follow:
- You can only write valid JSONs based on the documentation below:

{"from_address": "string", "to_address": "string"}

- Your goal is to transform email input into structured JSON. Comments are not allowed in the JSON. Text outside the 
JSON is strictly forbidden.
- Users provide the email freight orders as input, which you will transform into JSON format given the JSON 
documentation.
- You can enhance the output with general common-knowledge facts about the world relevant to the procurement event.
- If you cannot find a piece of information, you can leave the corresponding attribute as "".
```

### *Human message*

Message from a human. User input that the AI will process.

```text
Hello,
Please send me your offer for groupage transport for:
1 pallet: 120cm x 80cm x 120cm - weight approx 155 Kg
Loading: 300283 Timisoara, Romania
Unloading: 4715-405 Braga, Portugal
Can be picked up. Payment after 7 days
```

### *Response*

```json
{
  "from_address": "300283 Timisoara, Romania",
  "to_address": "4715-405 Braga, Portugal"
}
```
