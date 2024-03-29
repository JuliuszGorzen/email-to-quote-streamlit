### *System message*

Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
:green[{categories}] -> this will be replaced with the categories definitions

```text
For each text, mark NER tags. In [] mark the entity and in () mark the tag.
Tag categories:
{categories}
```

### *System message (categories)*

Definitions.

```markdown
| Tag Name             | Tag Definition                                                                                                                                                                                                                      |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| origin_location      | (Address) First pick-up location of transport. Should contain city and country at a minimum                                                                                                                                         |
| destination_location | (Address) Last delivery location of transport. Should contain city and country at a minimum                                                                                                                                         |
| date_of_mail         | (Date) Date when mail was sent / received                                                                                                                                                                                           |
| weight               | (Number) Load weight, Attribute “unit” contains one of supported weight units: Ton, Kilogram.If not specified assume 22t.                                                                                                           |
| start_of_transport   | (DateTime) Date & time when pick-up is to happen. If not specified, assume in 5 days at 7:00am UTC.                                                                                                                                 |
| num_activities       | (Number) Sum of the total number of pick-up & delivery locations. If not specified assume 2                                                                                                                                         |
| transport_mode       | (String) Required mode of transportation. If not specified, assume “ROAD”. Other options are “AIR”, “RAIL”, “MULTIMODAL”, “RIVER”                                                                                                   |
| load_type            | (String) Type of transport required. If not specified, assume Full Truck Load (FTL). Other options are Less-than Truck Load (LTL), Bulk. The later options usually apply if Note these can usually be inferred from the description |
| vehicle_type         | (String) Type of truck used for the transport. If not specified, assume “standard”. Other options are “Reefer”, “Tanker”, “Flatbed / Open platform trucks”                                                                          |
| hazardous_goods      | (String) Flag in case the transport contains hazardous goods. If not specified assume “No” 
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

```text
origin_location: [300283 Timisoara, Romania] (Address)
destination_location: [4715-405 Braga, Portugal] (Address)
weight: [155 Kg] (Number)
load_type: [groupage] (String)
start_of_transport: [7 days from now] (DateTime)
```
