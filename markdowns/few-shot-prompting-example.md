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

### *Human message 1*

Message from a human. Example 1.

```text
FROM: Juliusz Gorzen
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

I would like to book a FTL transport from Safranberg 123, 12345 Ulm to
Wietrzna 34, Wroclaw 52-023, Poland next Monday.

Thanks.

Sincerely,
Juliusz
```

### *AI message 1*

Message from a bot. Response to example 1.

```json
{
  "from_address": "Safranberg 123, 12345 Ulm",
  "to_address": "Wietrzna 34, Wroclaw 52-023, Poland"
}
```

### *Human message 2*

Message from a human. Example 2.

```text
FROM: Abc def
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

I would like to book a magic transport from 3486 Tuna Street, 48302, Bloomfield Township to
1011 Franklin Avenue, Daytona Beach 32114, US on 2024-03-03 10 pm.

Thanks you.

BR,
Abc
```

### *AI message 2*

Message from a bot. Response to example 2.

```json
{
  "from_address": "3486 Tuna Street, 48302, Bloomfield Township",
  "to_address": "1011 Franklin Avenue, Daytona Beach 32114, US"
}
```

### *Human message 3*

Message from a human. Example 3.

```text
FROM: John Taylor
RECEIVED: 2024-01-31 10:10:10.299064

Hi,

Pleas book the transport.
From: 2132 Thomas Street, Wheeling, US
To: -
Date: 2024-02-11 09:30
```

### *AI message 3*

Message from a bot. Response to example 3.

```json
{
  "from_address": "2132 Thomas Street, Wheeling, US",
  "to_address": ""
}
```

### *Human message 4*

Message from a human. User input.

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
