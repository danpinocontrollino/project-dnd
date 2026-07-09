---
license: cc-by-4.0
task_categories:
- text-generation
- text2text-generation
language:
- en
tags:
- story
- storytelling
- story generation
- dnd
- creative generation
- command generation
- dungeons and dragons
- ttrpg
- dungeon master
pretty_name: FIREBALL
language_creators:
- crowdsourced
source_datasets:
- original
size_categories:
- 100K<n<1M
paperswithcode_id: fireball
---

# Dataset Card for FIREBALL

## Table of Contents
- [Data Description](#data-description)
  - [DnD Turn Schema](#dnd-turn-schema)
    - [Normalized Actor State](#normalized-actor-state)
- [Additional Information](#additional-information)
  - [Citation](#citation)
  - [Licensing](#licensing)
  

---
## Data Description
**FIREBALL: A Dataset of Dungeons and Dragons Actual-Play with Structured Game State Information**

FIREBALL is a large crowdsourced dataset of people playing Dungeons and Dragons (D&D or DnD) on Discord. In addition to playing the game using natural language (primarily English), players also used a bot called [Avrae](https://avrae.io/). Avrae enables players to keep track of the state of the game by writing commands, which we collected.
This dataset contains nearly 25,000 unique sessions of gameplay, 153,829 turns, and detailed information about people's D&D game turns.


* [Published paper](https://aclanthology.org/2023.acl-long.229/)
* [Paper on arXiv](https://arxiv.org/abs/2305.01528)

**Abstract**
> Dungeons & Dragons (D&D) is a tabletop roleplaying game with complex natural language interactions between players and
> hidden state information. Recent work has shown that large language models (LLMs) that have access to state
> information can generate higher quality game turns than LLMs that use dialog history alone. However, previous work
> used game state information that was heuristically created and was not a true gold standard game state. We present
> FIREBALL, a large dataset containing nearly 25,000 unique sessions from real D&D gameplay on Discord with true game
> state info. We recorded game play sessions of players who used the Avrae bot, which was developed to aid people in
> playing D&D online, capturing language, game commands and underlying game state information. We demonstrate that
> FIREBALL can improve natural language generation (NLG) by using Avrae state information, improving both automated
> metrics and human judgments of quality. Additionally, we show that LLMs can generate executable Avrae commands,
> particularly after finetuning.



**Note:** This dataset requires the `jsonlines` library to be imported.


### DnD Turn Schema

Each line of the dataset contains a filtered schema for each conversational turn.
The schema includes the following keys:
```
{
    "speaker_id": The anonymized user ID of the user who sent the commands in the triple. 
    "before_utterances": A list of strings corresponding to the "preceding" utterances in the triple.
    "combat_state_before": A list of normalized actor states (see below) for each actor in the combat instance at the instant before the command was run.
    "current_actor": (nullable) The normalized actor state of the actor whose turn it currently is.
    "commands_norm": A list of strings corresponding to the "commands" portion of the triple.
    "automation_results": A mechanically generated list of strings representing the results of running the action in the Avrae engine.
    "caster_after": The normalized actor state of the actor who ran the action(s), which may or may not be the current actor.
    "targets_after": A list of normalized actor states for each actor who was targeted by the action.
    "combat_state_after": A list of normalized actor states for each actor in the combat instance at the instant after the command was run.
    "after_utterances": A list of strings corresponding to the "following" utterances in the triple.
    "utterance_history": The last 5 messages in the chat history before the command was run.
    "before_idxs": A list of integers corresponding to the index of the "message" events containing the "preceding" utterances in the raw event file.
    "before_state_idx": The index of the "combat_state_update" event in the raw event file that was used to derive "combat_state_before".
    "command_idxs": The indexes of the "command" events corresponding to the "commands_norm" key.
    "after_state_idx": The index of the "combat_state_update" event corresponding to the "combat_state_after" key.
    "after_idxs": The indexes of the "message" events corresponding to the "after_utterances" key.
    "embed_idxs": (nullable, same length as "automation_results") The indexes of "message" events corresponding to rich results shown to players on Discord for each result in the "automation_results" key.
}
```
All user IDs and usernames have been randomized (by way of a hash function) to preserve anonymity.

#### Normalized Actor State
The normalized actor state is only a subset of the available actor information, corresponding to the information we used for our engineering experiments for the FIREBALL paper. For a full list of available actor information, see table 6 in the [FIREBALL paper](https://aclanthology.org/2023.acl-long.229/).
```
{
    "name": The name of the actor.
    "hp": The numerical and narrative hit points (e.g. "<12/34; Bloodied>").
    "class": The actor's class(es) and level(s), if applicable (e.g. "Fighter 3")
    "race": The actor's race, if applicable (e.g. "Mountain Dwarf", "Adult Red Dragon").
    "attacks": A list of the actor's available attack names.
    "spells": A list of the actor's available spells.
    "actions": A list of the actor's available special abilities.
    "effects": A list of any temporary effects on the actor (e.g. "Stunned").
    "description": The actor's narrative description (if available).
    "controller_id": The anonymized user ID of this actor's controller.
}
```
`combat_state_before`, `current_actor`, `caster_after`, `targets_after`, and `combat_state_after` use the above state format.

## Additional Information
### Citation
```
@inproceedings{Zhu2023FIREBALL,
title={{FIREBALL: A Dataset of Dungeons and Dragons Actual-Play with Structured Game State Information}},
author={Zhu, Andrew and Aggarwal, Karmanya and Feng, Alexander and Martin, Lara J. and Callison-Burch, Chris},
year={2023},
booktitle={Annual Meeting of the Association for Computational Linguistics (ACL)},
month={7},
url={https://aclanthology.org/2023.acl-long.229/},
address={Toronto, Canada},
pages={4171--4193},
publisher={ACL},
doi={10.18653/v1/2023.acl-long.229}
}
```
---
### Licensing
The Creative Commons Attribution 4.0 International License. https://creativecommons.org/licenses/by/4.0/