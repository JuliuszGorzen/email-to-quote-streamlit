### *System message*

Message for priming AI behavior, usually passed in as the first of a sequence of input messages.
:green[{context}] -> this will be replaced with the uploaded file

```text
You are a bot that answers the following question based only on the provided context:

<context>
{context}
</context>
```

### *Uploaded file*
```markdown
# Autonomus Quotation Entity Definitions
This document represents the entity definitions for the Autonomus Quotation API that are needed to be able to return the quote.

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
| hazardous_goods      | (String) Flag in case the transport contains hazardous goods. If not specified assume “No”                                                                                                                                          |
```

### *Human message*

Message from a human. User input that the AI will process.

```text
Which entities are related to the location. List all with the descriptions.
```

### *Response*

```text
The entities related to the location are:

1. origin_location: Represents the first pick-up location of the transport. It should contain at least the city and country information.

2. destination_location: Represents the last delivery location of the transport. It should contain at least the city and country information.
```
