# The best ideas of the legacy model

> **Status (Nov 2026):** catalogue only. No code was changed.
>
> **Decision.** The new implementation starts from a fresh repository. It keeps
> none of the legacy code, not even as a baseline. This document preserves what
> is worth keeping: the *ideas*. It covers the whole model, not just memory:
> - population, geography and demography;
> - families, networks and conversations;
> - institutions and policy levers;
> - learning, motivation and outputs.
>
> **Sources.** The legacy code (read in full for this document),
> [`legacy_language_model_analysis.md`](../legacy_language_model_analysis.md),
> [`memory_model_analysis.md`](../memory_model_analysis.md) (**MMA**),
> [`learning_model_new_approaches.md`](../learning_model_new_approaches.md)
> (**LMNA**) and the companion documents in this folder. Code references such as
> `agent.py:1522` point to the legacy repository
> (`github.com/pgervila/bilingual_society_ABM_legacy`) at commit `b23edcd`. They
> are there for traceability only.

---

## 0. Summary

The legacy model's core insight is still right. **Language shift is the
cumulative result of millions of ordinary conversations.** Who talks to whom,
where, in which language and how often decides what each person learns and
forgets. What people know then decides the next conversations, which languages
they pass to their children, and which jobs and friends they get.

The legacy code implemented this loop with weak parts:
- a compressed vocabulary;
- an additive memory rule;
- exposure bundled into 10-day calls;
- unmeasurable thresholds.

**The loop itself, and most of the social mechanisms around it, are worth
carrying forward.**

The twelve most valuable ideas:

| # | Idea | Section |
|---|---|---|
| 1 | Bilingualism and shift **emerge** from conversations between individuals; nothing at the macro level is programmed | §2.1 |
| 2 | A **life course** with stage-specific daily routines (home, daycare, school, university, work, retirement) that decides each person's exposure | §2.2 |
| 3 | Towns of **heavy-tailed sizes** with a tunable **language geography** (segregated or mixed) | §3.1 |
| 4 | **Intergenerational transmission as an explicit choice**: each parent picks the language to speak to the newborn, weighted by proficiency, and the wider family follows | §4.1 |
| 5 | **Relationship-specific language memory**: each dyad remembers the language it used, and it sticks | §5.2 |
| 6 | **MAXIMIN language choice** with receptive multilingual conversations and the muting of people who cannot follow | §6.2 |
| 7 | Conversation **length and register** depend on mutual intelligibility, age, and the youngest listener | §6.3 |
| 8 | **School language policy** limited by the supply of teachers, and **job language requirements** derived from the local population | §7 |
| 9 | Knowledge **per word**, with decay through disuse, **speaking reinforcing more than listening**, and code-switching **emerging from failed retrieval** | §8 |
| 10 | **Linguistic exclusion** is remembered and triggers **motivated learning** | §9 |
| 11 | **Initial conditions built by simulating life histories**, correlated within families | §10.1 |
| 12 | Outputs that track **exposure** (tokens heard and spoken per language) as well as knowledge | §10.2 |

---

## 1. How to read this document

Each section lists ideas in a table with four columns:
- **Idea:** the mechanism, in one line.
- **In legacy:** what the code did, with a reference. The status is
  *implemented*, *partly* (implemented with a known flaw or gap) or *sketched*
  (TODO, stub or an unmerged 2017 branch).
- **Why keep it:** the empirical or theoretical reason.
- **In the new model:** how the idea should be rebuilt. "Exposure model",
  "virtual tests", "timestamped events" and approach "A" refer to LMNA;
  "F1–F5" refer to
  [`essential_language_features.md`](essential_language_features.md).

Section 12 lists what *not* to carry over, so the flaws are not copied along
with the ideas.

---

## 2. Architecture and time

### 2.1 The causal loop

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Shift emerges from conversations | All learning comes from `run_conversation` (`model.py:287`) and study; the population's language mix is an output, never an input rule (implemented) | Macro models of language competition (Abrams & Strogatz, 2003) treat speakers as states. An ABM's value is in getting individual trajectories right and letting the macro pattern emerge. | Keep as the founding principle. Each link of the loop (exposure → knowledge → decision) is validated separately against data. |
| Knowledge feeds back into social structure | Proficiency decides conversation language, newborn language, friendships, partner choice and hiring (implemented) | This feedback is what makes shift self-reinforcing or self-limiting | Keep every feedback path, but have decisions read **measurable proficiency** (coverage, productive success) instead of `pct` |
| Agents reclassify their own language profile | `update_lang_switch` (`agent.py:265`) moves an agent between monolingual and bilingual as knowledge changes (implemented) | Language profile is a state that changes over a life, not a birth label | Keep a categorical profile as a **census-like output** derived from proficiency. Decisions should read the graded measures, not the 3-way label. |

### 2.2 Life course and daily routine

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Life stages with their own routines | Agent classes `Baby` (0–2) → `Child` (2–12) → `Adolescent` (12–18) → `Young`/`YoungUniv` (19–30) → `Adult` (30–65) → `Pensioner`, plus `Teacher`/`TeacherUniv` (`agent.py:1266–2245`) (implemented) | Exposure depends heavily on life stage: home-centred infancy, school, university, work, retirement | Keep the stages, but as **data** (a life-stage field plus an activity schedule), not as a class hierarchy whose instances are rebuilt on every transition |
| A time budget of activities | Four stages per step (home, school or work, school or work, home and leisure); each activity has a frequency `num_days` out of 10 (partly) | It captures how often each setting recurs, which is the right concept | Rebuild as a **weekly time-use schedule** (school or work days vs weekends) calibrated on time-use surveys. Each occurrence becomes a **timestamped event**. Bundling `num_days` into one call was the flaw (MMA M2, M12). |
| Random activation order within a step | `shuffle=True` in the scheduler (`schedule.py`) (implemented) | Avoids artefacts from a fixed order | Keep. Make results invariant to order by applying learning at event timestamps. |
| Networks frozen within a step | Adjacency matrices snapshot once per step (`schedule.py`, `networks.py:222`) (implemented) | Consistency: a step's decisions see one state of the world | Keep. Apply network updates between steps. |

---

## 3. Population, geography and demography

### 3.1 Towns and language geography

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Towns of heavy-tailed sizes | Cluster sizes drawn from a Pareto distribution (shape 1.25), minimum 100 people (`geomapping.py:97`) (implemented) | Town sizes follow heavy-tailed distributions, close to Zipf's law for cities (Gabaix, 1999) | Keep. For Catalonia, use real municipality sizes (Idescat) or a distribution fitted to them. |
| Language geography as a scenario parameter | The initial language labels can be segregated by distance, sorted within towns, or mixed in family-coherent blocks (`geomapping.py:370`) (implemented) | Spatial segregation is a classic determinant of minority-language survival. Making it a parameter lets scenarios test it. | Keep. Calibrate the initial geography on census knowledge by municipality. |
| Places on a map | Homes, schools, universities (about 20 % of towns) and jobs placed in each town; agents move between them (implemented) | Co-presence in places is what makes conversations possible | Keep places as **settings with a register and a language mix** (F3). A grid is optional; what matters is who shares which setting. |

### 3.2 Demography

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Realistic mortality | A Gompertz-type death curve fitted with TOPALS (de Beer, 2012): life expectancy about 77, standard deviation about 15 (`model.py:232`) (implemented) | Population turnover drives intergenerational change | Keep, with cohort life tables (INE, Idescat) |
| Couples formed from acquaintances | Partner search among people met more than 10 times, at most 10 years apart in age, compatible language profiles, each knowing the common language above a threshold; a hazard of about 1 per 6 years for `Young`, 1 per 4 for `Adult`; only with a job (`agent.py:1522–1583`) (implemented) | Couples form within social circles, and the couple's common language matters for the home language. Mixed couples are where transmission is decided. | Keep partner choice from acquaintances, with the shared language as a factor. Calibrate the rate of linguistically mixed couples on census or EULP data, and do not hard-code who can partner whom. |
| Fertility of couples | A per-step birth probability for married agents, up to 4 children, until 40 (`agent.py:1584`, `:1962`) (implemented) | Fertility differences between language groups affect shift | Keep. Use age-specific fertility rates (INE). |
| Immigration as an exogenous driver | Families of 4 arriving yearly at a rate (0.5 % of the population), placed in towns in proportion to size, hired regardless of language (`model.py:585–617`) (implemented) | Large immigration waves (in Catalonia 1950s–70s, and since 2000) are a central driver of language contact | Keep. Give immigrants an origin and language profile (Spanish-speaking, other languages), and calibrate flows on migration statistics, including cohort history (review N20). |
| Residential mobility tied to jobs and marriage | Couples move (near the job, half-way between two jobs, or to the town with more jobs); the family follows a job move and children change school; students move to university towns into shared flats (`agent.py:1667–1791`, `:1893`) (implemented) | A move changes a person's networks and exposure. It is a natural experiment the model can reproduce. | Keep. Calibrate on internal-migration data. Moves are the standard attrition scenario (V4, V5). |

---

## 4. Families and intergenerational transmission

### 4.1 Transmission

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Each parent chooses the language to speak to the newborn | `get_newborn_lang` (`model.py:247`): a monolingual parent speaks their language; a bilingual parent chooses with probability proportional to their knowledge of each language (implemented) | Intergenerational transmission is the decisive stage for a language's survival (Fishman, 1991, stage 6). Modelling it as each parent's choice is the right granularity. | Keep it as a **calibrated decision**. EULP reports the language people spoke with their parents and the language they speak with their children, so transmission rates can be fitted by couple type. Add the factors the code's TODOs name: consistency with older siblings, the couple's language, the local environment. |
| The child's profile follows from what parents speak to it | Newborn type is derived from the set of parental languages (implemented) | — | Replace the label with **real input**: the child learns from what it actually hears (directed and overheard, review N18). Its profile is an output. |
| The wider family follows the parents' choice | `set_newborn_family_links` (`networks.py:138`): grandparents, uncles, aunts and cousins speak to the baby in the language its linking parent uses, if they speak it well enough (implemented) | Family language norms spread through kinship. Grandparents are a major source of minority-language input. | Keep, as a default that relatives' own proficiency can override |
| Typed, extended kinship | The family network has typed and reciprocal relations: consort, parent/child, grandparent/grandchild, uncle or aunt/nephew, sibling, cousin (`networks.py:17`) (implemented) | Different relatives give different amounts of input | Keep |
| Family gatherings led by the elderly | Pensioners occasionally gather children, their partners and grandchildren; the grandchildren talk among themselves (`agent.py:2178`) (implemented) | Captures grandparent–grandchild contact and multigenerational talk at family events | Keep, with frequency calibrated on time-use data |
| Family-correlated initial knowledge | Children's initial bilingual level is derived from their parents' levels (`model.py:480`) (implemented) | Knowledge is correlated within households | Keep the idea. In the new model it falls out of simulating each family's history (§10.1). |

---

## 5. Social networks

### 5.1 Network types

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Typed networks | Four graphs: acquaintances (directed), friendship, family (typed) and links between workplaces (`networks.py:31`) (implemented) | Relationships differ in frequency, intimacy and language norms | Keep. Add tie strength to every type. |
| Acquaintances accumulate through meetings | Each conversation adds or strengthens a directed acquaintance edge with a meeting count (`agent.py:965`) (implemented) | Familiarity grows with repeated contact. It is the pool from which friends and partners come. | Keep |
| Friendships from repeated shared context | Friends are chosen from classmates or colleagues and from acquaintances met at least 10 times. Constraints: a heavy-tailed capacity for friends (lognormal, at least 3), compatible language profiles, and similar proficiency in the common language. Updated every 2 years (`agent.py:1052–1096`, `model.py:569`) (implemented) | Friendship forms through repeated contact in shared settings (focus theory) and shows homophily, including linguistic homophily | Keep. Calibrate network size and homophily on survey data (EULP asks the language spoken with friends). |
| Tie strength | Friendship edges carry a random intensity of 1–10, and friends are picked in proportion to it (`agent.py:1092`, `:1207`) (implemented) | Exposure concentrates on a few strong ties | Keep, with intensities drawn from a measured distribution |
| Service encounters across workplaces | A random graph between workplaces; workers talk to a random employee of a linked workplace (`networks.py:130`, `agent.py:1814`) (implemented) | Contact with strangers (shops, services, customers) is where language choice is most contested. EULP measures language use in shops and with strangers. | Keep, as a "services" setting with its own language-choice norms |

### 5.2 Language memory in relationships

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Each dyad remembers its language | Every acquaintance edge stores the language used; later conversations reuse it (`model.py:382`, `agent.py:469`) (implemented) | People tend to keep the language they first used with someone, so language choice has inertia at the level of the relationship | Keep. Make the inertia a parameter, and let it **renegotiate** when one partner's proficiency changes. A TODO already proposes this (`agent.py:276`, `:755`). Calibrate on EULP's language-by-interlocutor data. |
| A group's language from its members' histories | For a group of known people, the language is the rounded average of the languages used with each (`model.py:390`) (implemented) | Group norms come from dyadic norms | Keep the principle. Replace rounding with a probabilistic choice. |

---

## 6. Conversations and language choice

### 6.1 The conversation as the unit of exposure

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Conversations with roles | An initiator and others; every non-muted participant speaks in turn and all others listen (`model.py:287`) (implemented) | Speaking and listening have different effects on memory | Keep. Record each event (who, setting, language, words, timestamp) for the learner. |
| Group conversations | `speak_in_random_subgroups` (`agent.py:984`) picks a random subgroup of classmates or colleagues, then one person; teachers address the whole class (implemented) | Real talk happens in groups whose composition decides the language | Keep. Calibrate group sizes per setting. |
| Bystanders | `listen()` without a target overhears a conversation between two others in the same place (`agent.py:444`) (implemented) | Overheard speech is part of input, though it predicts children's vocabulary less than speech addressed to them (Weisleder & Fernald, 2013) | Keep, with **separate weights for directed and overheard input** (review N18) |

### 6.2 MAXIMIN and accommodation

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| MAXIMIN language choice | `get_conv_params` (`model.py:338`) chooses by the language profiles present: monolinguals of one side plus bilinguals → that side's language; all bilingual → the initiator's dominant language (implemented) | Van Parijs (2011): groups converge on the language best known by the least competent participant. It matches the Catalan **accommodation norm**: Catalan speakers switch to Spanish with people they take to be Spanish speakers (Woolard, 1989). | Keep as the default rule, with its parameters (how strongly speakers accommodate, by setting) calibrated on EULP |
| Receptive multilingual conversations | When everybody understands both languages but not everyone speaks both, each speaks their preferred language (`model.py:414–425`) (implemented) | Receptive multilingualism, where each person speaks their own language and understands the other's, is common between closely related languages (ten Thije & Zeevaert, 2007), and in Catalonia | Keep. It depends on comprehension of the unlearned language, which F2 makes realistic through form overlap. |
| Muting and exclusion | People who cannot follow are muted; their exclusion is counted in the language they lack (`model.py:426–470`, `agent.py:1005`) (implemented) | Being unable to take part is a strong motive to learn, and a cost the model can measure | Keep (§9) |
| Who starts matters | The initiator's dominant language decides among bilinguals; for incompatible monolinguals the initiator speaks to whoever understands (implemented) | Opening moves set the language of an interaction | Keep. TODOs add place bias, "those who can should adapt to the initiator's language", and recording the language the initiator intended (`model.py:394–418`). |

### 6.3 Length and register

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Intelligibility shortens conversations | Multilingual conversations are short; conversations with muted participants are very short (implemented) | Communication difficulty reduces interaction, and with it input | Keep, with length depending on measured comprehension |
| An age curve for talk | Talk per conversation rises through childhood, plateaus, and declines after 65 (`model.py:195`) (implemented) | The amount of speech varies with age | Keep. Calibrate on words-per-day norms (Mehl et al., 2007; LENA for children). |
| Speech adapted to the youngest listener | Speakers sample words from the youngest listener's age distribution (`min_age_interlocs`, `agent.py:893`) (partly) | Child-directed speech is a distinct register | Keep, as **registers** (child-directed, conversational, school, work, media) with their own word frequencies (F1, F3). The flaw was that it reinforced adults' rarest slots (MMA M8). |

---

## 7. Institutions and policy levers

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| School language policy as a lever | Each school has a policy: the set of teacher profiles allowed (`[0, 1]` L1-medium, `[1]` bilingual, `[1, 2]` L2-medium). The policy also sets adolescents' study time per language (`agent.py:1482–1490`) (partly) | Medium-of-instruction policy is the main policy lever in bilingual societies (in Catalonia, Catalan-medium immersion) | Keep. Define policy as **hours of instruction per language per grade**, plus the language of school administration and playground norms. Note that the teacher's language in class was hard-coded to L1 (§12). |
| Policies limited by the teacher workforce | Teachers are hired from the town, then other towns, then recruited from companies, according to the language profiles the policy requires. At setup the model stops if a policy cannot be staffed (`geomapping.py:474`); during a run, missing teachers are created from nothing (`city_objects.py:196`) (partly) | A real constraint: a policy needs enough qualified teachers | Keep. Make staffing shortfalls an output (a measurable cost of a policy), neither a crash nor teachers created from nothing. |
| School organisation | Courses by age; yearly promotion; teacher reassignment, a swap every 4 years, retirement backfill; universities with faculties (`city_objects.py:85–186`, `:585`) (implemented) | Classmates and teachers form the child's main non-family network | Keep the structure. Calibrate class sizes and teacher turnover. |
| Daycare and the parent–teacher interface | Babies over 1 year attend daycare; parents talk with teachers and other parents at drop-off and pick-up (`agent.py:1307–1351`) (implemented) | Early childcare is an input channel, and a meeting place for parents | Keep. Remove the "no input before 1 year" gate (MMA M13). |
| Job language requirements from local demography | A workplace requires the languages of every group above 10 % of its town, collapsing to "bilingual only" when both monolingual groups are large (`city_objects.py:811`) (implemented) | Labour-market requirements are a strong incentive to learn a language | Keep the mechanism (requirements depend on the local clientele), with **sector-specific** policies and thresholds calibrated on EULP data on language at work |
| Hiring tied to language, with a reaction to rejection | Hiring fails on a language mismatch and triggers a reaction to exclusion (`city_objects.py:897`) (implemented) | Economic incentive is a classic driver of second-language learning and of shift | Keep, with explicit costs and benefits (§11, the 2017 incentives idea) |
| Job search with local bias and family constraints | The own town is searched more than half the time; moving town needs 3 years in the current job and a partner free to move (`agent.py:1608`, `city_objects.py:832`) (implemented) | Realistic frictions on mobility | Keep. Calibrate on labour-mobility data. |
| Media as an exogenous channel | `listen_to_media` is a stub; a 2017 branch consumed media with a 20 % probability (sketched) | Media is an exogenous input and a policy lever (LMNA §6.3) | Build it: media exposure per language, by age, from media-consumption surveys |

---

## 8. Language knowledge and learning

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Knowledge per word, not one proficiency number | Each person has per-word memory arrays for each language (implemented, on compressed slots) | It allows domain profiles, coverage, vocabulary tests and code-switching | Keep, on **real lemmas** (F1) |
| Memory decays with disuse and strengthens with use | Stability and retrievability (`S`, `R`), from the SuperMemo line of models (implemented, with the wrong rule) | A memory trace per item is the standard way to model vocabulary retention | Keep the trace concept. Replace the rule (MMA §4; LMNA §4.3; candidate rules compared at decision point D1). |
| Speaking reinforces more than listening | `ds_factor` 1.0 vs 0.1 (implemented) | Retrieving a word strengthens it more than hearing it (testing and generation effects; review N7) | Keep. **Estimate** the ratio instead of asserting it. |
| A graded hierarchy of encoding by engagement | A TODO proposes reading < media < listening in conversation < speaking (`agent.py:699`) (sketched) | More engagement, more encoding | Keep, as **encoding weights per event type** fitted to data (LMNA §4.3, `ω`) |
| New words need repeated exposure | The encoding threshold `mem_eff` (3–6 encounters) (`agent.py:200`) (partly) | Incidental learning needs repeated encounters (Webb, 2007; Uchihara et al., 2019) | Keep the finding. Make the mechanism inference of meaning plus a memory threshold (LMNA §4.5). |
| Comprehension gates learning | The chance of picking up an unknown word rises with the share of the conversation understood (`agent.py:563`) (implemented) | Learning from context needs enough coverage (Hu & Nation, 2000) | Keep (LMNA §4.5, `p_inf`) |
| Cognates help | Unknown words resembling a known word in the other language are recognised (implemented, with random distances) | Cognate facilitation is robust (Costa et al., 2000) | Keep, with **real form distances** (F2) |
| Code-switching emerges from failed retrieval | The cascade L1 → L12 → L2 in `pick_vocab` (`agent.py:859`) (partly) | No separate switching module is needed | Keep the principle: competition between translation equivalents, reinforcing the word actually used (F4). Drop the `L12`/`L21` stores (§12). |
| Borrowing habits | The `L12`/`L21` stores held "what I usually say in the other language" (partly) | Established loans exist (*vale*, *bueno* in Catalan) | Rebuild as the **variant layer** (F5; review N8) |
| A shared conceptual store | The author's TODO: "known thoughts are the UNION of all words known in L1 + L12 + L21 + L2" (`agent.py:889`); the same concept index is used for both languages (sketched / implemented) | Bilinguals share concepts across languages (RHM, BIA+) | Keep: concept nodes shared between languages, with language-specific lemmas (F4) |
| Frequency-structured word sampling | A three-stage Zipf–Mandelbrot fitted to the British National Corpus (`Zipf.py`) (implemented) | Word frequencies follow a piecewise power law | Replace with **measured frequencies** per register (SUBTLEX-CAT/ESP, CHILDES). The fitted curve is still useful for synthetic tests. |
| Vocabulary grows with age | An age-indexed vocabulary ceiling (implemented as a sampling ceiling) | Children's lexicons grow by orders of magnitude | Keep the finding. Growth should emerge from exposure and memory, not from a ceiling (F1). |
| Self-study as an exposure mode | `study_vocab` (`agent.py:835`) (implemented) | Schoolwork and reading are input | Keep, inside the literacy stage (LMNA A2) |
| Memory declines in old age | `S` reduced after 65 in the IC recipe; `Pensioner.update_mem_decay` is defined but never called (`agent.py:2213`) (sketched) | Retrieval failures increase with age (Burke et al., 1991) | Keep as a fitted age effect on retrieval (LMNA §4.4) |

---

## 9. Motivation and reaction to exclusion

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Exclusion is remembered | Being muted increments a per-age exclusion counter in the missing language (`agent.py:1005`) (implemented) | Repeated communicative failure is an experience that accumulates | Keep |
| A short memory for exclusion | An exponentially weighted moving average over the last 5 steps; above 1, the agent reacts (`agent.py:1011`, `:223`) (implemented) | Motivation reacts to *recent* experience | Keep the moving-average idea, with a timescale calibrated in days |
| Exclusion triggers learning | The reaction is to study vocabulary, or a teacher explains a word (`agent.py:1029`, `:1236`) (partly: one word or one study session) | Motivation drives second-language learning (Gardner, 1985) | Keep, with richer reactions, as the TODOs list (`agent.py:1246–1249`): study, ask friends, join conversations in the language, consume media, and eventually change networks (legacy analysis, suggestion 6) |

---

## 10. Initial conditions and outputs

### 10.1 Initial conditions

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Initial knowledge from simulated life histories | The IC recipe simulates one speaker from birth at a constant exposure share and snapshots every age (`ic_generator.py`) (implemented, with a different rule from the live model) | Initial agents should know what their history would have taught them | Keep: **spin-up with the same learner** that runs the simulation (LMNA §3.7). The exposure histories must be specific to each cohort (review N20). |
| Exposure levels | Six exposure buckets (10–100 %) for bilingual profiles (implemented) | A simple way to span bilingual profiles | Keep as scenario archetypes for the test harness (LMNA §3.3) |
| Daily volume by age | `words_day_factor`: infants hear little; adult plateau; decline after 65; own speech, TV and radio counted (`ic_generator.py:57`) (implemented) | Input volume changes with age | Keep the idea. Anchor the volumes on Mehl et al. (2007) and LENA norms. |
| Content-addressed cache and RNG isolation | The IC table cache is keyed by a hash of its inputs; the generator never touches the global RNG (`ic_generator.py:201`) (implemented) | Results must not depend on cache state or call order | Keep both as engineering conventions (LMNA §3.9) |

### 10.2 Outputs

| Idea | In legacy | Why keep it | In the new model |
|---|---|---|---|
| Population shares by language profile | Shares of L1-monolinguals, bilinguals and L2-monolinguals over time, by town (`dataprocess.py`) (implemented) | The headline outcome of shift | Keep, as census-like categories (understands, speaks, reads, writes) from the virtual test battery |
| Exposure tracking | Tokens processed per step and language per agent (`tokens_per_step_*`), conversations per step (implemented) | Exposure is the input to learning, and it can be measured in people (LENA, experience sampling) | Keep, in real words per day, per setting and per language. This is what calibrates the exposure model. |
| Separate levels for understanding, speaking and bilingualism | Three thresholds: 0.025, 0.1 and 0.2 (implemented, in the wrong units) | The distinction between understanding and speaking is right | Keep the three levels, defined in measurable units (LMNA §3.1, rule 3) |
| Results in parts | Parquet parts flushed periodically to bound RAM (implemented) | Long runs produce a lot of data | Keep (LMNA §3.9) |

---

## 11. Ideas sketched but never finished

These come from TODO comments and from 2017 branches that were never merged
(`model_lang_incentives`, `model_mass_media_influence`, `model_shops`). Each
deserves a decision in the new design.

| Idea | Source | Assessment |
|---|---|---|
| **Language pride vs financial interest**: agents weigh identity against economic payoff (`lang_pride`, `financial_greed`, wealth) | branch `model_lang_incentives` (2017) | **Important.** Attitudes and identity are probably the strongest drivers of shift (§8 of the features document). Add attitude variables with measurable counterparts (EULP attitude items). |
| **Mass media** consumption per language | branch `model_mass_media_influence`; `listen_to_media` stub | **Important.** An exogenous channel and a policy lever (§7) |
| **Shops and meeting points** as places of contact | branch `model_shops`; `MeetingPoint`/`Store` stubs | Useful: the "services" setting (§5.1) |
| **Foreigner talk**: native speakers adapt to learners | TODO at `agent.py:882` (Ferguson, 1975) | Useful: accommodation of register, which changes the input learners get |
| **Monolinguals react to mixing**: surprise as corrective feedback | TODO at `agent.py:903` | Useful: normative pressure against code-switching and loans (F5) |
| **Place bias in language choice** | TODO at `model.py:401` | Useful: settings with their own language norms (F3) |
| **Renegotiating a dyad's language** when proficiency changes | TODOs at `agent.py:276`, `:755` | Needed for realistic stickiness (§5.2) |
| **Consistency of transmission across siblings**; home language drift | TODO at `model.py:258`; legacy analysis, suggestion 7 | Needed for transmission (§4.1) |
| **Small-world networks; friends of friends** | TODOs at `networks.py:101`, `agent.py:1503` | Useful: network closure and clustering |
| **Job search weighted by distance and town language** | TODO at `agent.py:1617` | Useful: language-driven mobility |
| **Teachers hired by competence**, with higher requirements for older students | TODOs at `city_objects.py:280–281`, `:325` | Useful for school policy (§7) |
| **Recording the intended vs the realised language** | TODO at `model.py:394` | Useful: a measurable output for language-choice studies |
| **An endogenous economy** (job seats, wealth) | `city_objects.py:946`; 2017 wealth idea | Later. It needs its own validation. |

---

## 12. What not to carry over

Copying an idea should not copy these flaws. Each is documented elsewhere.

| Flaw | Where documented |
|---|---|
| 500 compressed slots whose meaning shifts with age | MMA M8; LMNA S1 |
| The SuperMemo rule applied additively; repetitions within a call double-counted; no lapses; single-exponential forgetting | MMA M1–M3, M5 |
| Exposure bundled into 10-day calls (`num_days`); results that depend on how calls are split | MMA M2, M12 |
| `pct` (types with `R > 0.9`) and the thresholds 0.025 / 0.1 / 0.2 / 0.3 | MMA M7; LMNA S12 |
| Random edit distances as "similarity" | MMA M11 |
| `L12`/`L21` stores counted as knowledge of the other language | MMA M4 |
| **The teacher always speaks L1 in class** (`speak_to_group(studs, lang=0, …)`, `agent.py:2053`, `:2131`), so school language policy acts only through which teachers are hired and through adolescents' study time | new finding |
| No input for babies before 1 year | MMA M13 |
| Initial conditions built with a different rule from the live dynamics | MMA M13 |
| Life stages as classes whose instances are rebuilt on every transition | §2.2 |
| Hard-coded partner rules | §3.2 |
| Bilingual children of L2-monolingual parents given a 90 % L1 profile | `docs/components/model.md` |
| Seeds drawn at module load; conversations aborted on `ZeroDivisionError`; agents scheduled with `pos is None` | CLAUDE.md; MMA Appendix A |

---

## 13. Where each idea goes in the new design

| New component (LMNA §3) | Legacy ideas it absorbs |
|---|---|
| **Population and demography** | heavy-tailed towns and language geography (§3.1); mortality, couples, fertility, immigration, mobility (§3.2); typed kinship (§4.1) |
| **Settings and schedule** | life stages and routines (§2.2); places; schools, daycare, universities, jobs, services, media (§7, §11) |
| **Networks** | typed networks, acquaintance build-up, friendship from shared context, tie strength (§5.1) |
| **Language-choice layer** | dyadic language memory (§5.2); MAXIMIN, receptive multilingualism, muting, initiator effects (§6.2); place bias and renegotiation (§11) |
| **Exposure model** | conversation roles, groups, bystanders (§6.1); length and register (§6.3); daily volumes by age (§10.1) |
| **Learning core** (approach A) | per-word knowledge, decay and reinforcement, speaking vs listening, graded encoding, comprehension-gated learning, cognates, emergent code-switching, a shared concept store (§8) |
| **Motivation layer** | exclusion memory and reactions (§9); pride vs interest (§11) |
| **Transmission** | parental language choice, the wider family following it (§4.1) |
| **Initial conditions** | spin-up from life histories, cohort-specific (§10.1) |
| **Measurement** | census-like categories, exposure tracking, three levels of knowledge (§10.2) |

**The principle for every row:** keep the mechanism, express it in measurable
quantities, and calibrate its parameters on data:
- EULP, the census, and time-use and migration statistics for the social
  layer;
- the test battery for learning.

None of it is validated against legacy output.

---

## References

References already listed in LMNA Appendix C, the review, or the companion
documents (e.g. Abrams & Strogatz, 2003; Costa et al., 2000; Fishman, 1991;
Hu & Nation, 2000; Mehl et al., 2007; Uchihara et al., 2019; Webb, 2007;
Weisleder & Fernald, 2013) are not repeated.

- Burke, D. M., MacKay, D. G., Worthley, J. S., & Wade, E. (1991). On the tip of the tongue: what causes word finding failures in young and older adults? *Journal of Memory and Language*, 30, 542–579.
- de Beer, J. (2012). Smoothing and projecting age-specific probabilities of death by TOPALS. *Demographic Research*, 27(20).
- Ferguson, C. A. (1975). Toward a characterization of English foreigner talk. *Anthropological Linguistics*, 17(1).
- Gabaix, X. (1999). Zipf's law for cities: an explanation. *Quarterly Journal of Economics*, 114(3).
- Gardner, R. C. (1985). *Social Psychology and Second Language Learning: The Role of Attitudes and Motivation*. Edward Arnold.
- ten Thije, J. D., & Zeevaert, L. (eds.) (2007). *Receptive Multilingualism: Linguistic Analyses, Language Policies and Didactic Concepts*. John Benjamins.
- Van Parijs, P. (2011). *Linguistic Justice for Europe and for the World*. Oxford University Press.
- Woolard, K. A. (1989). *Double Talk: Bilingualism and the Politics of Ethnicity in Catalonia*. Stanford University Press.
