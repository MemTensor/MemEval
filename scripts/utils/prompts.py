"""Centralized prompt templates for supported benchmarks and memory libraries.

Organized into four sections:
  1. Shared context templates     – used across benchmarks and libs
  2. Lib-specific prompts         – tied to a particular memory framework
  3. Benchmark answer prompts     – per-benchmark QA generation
  4. Benchmark judge prompts      – per-benchmark LLM-as-judge evaluation
"""

# ── 1. Shared context templates ──────────────────────────────────────────────

CONTEXT_TEMPLATE = "Conversation memories:\n\n{memories}"

DUAL_SPEAKER_TEMPLATE = """Memories for user {speaker_1}:

    {speaker_1_memories}

    Memories for user {speaker_2}:

    {speaker_2_memories}
"""


# ── 2. Lib-specific prompts ─────────────────────────────────────────────────

MEMOS_CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


# ── 3. Benchmark answer prompts ──────────────────────────────────────────────

# -- LoCoMo --

LOCOMO_ANSWER_PROMPT = """
    You are a knowledgeable and helpful AI assistant.

   # CONTEXT:
   You have access to memories from two speakers in a conversation. These memories contain
   timestamped information that may be relevant to answering the question.

   # INSTRUCTIONS:
   1. Carefully analyze all provided memories. Synthesize information across different entries if needed to form a complete answer.
   2. Pay close attention to the timestamps to determine the answer. If memories contain contradictory information, the **most recent memory** is the source of truth.
   3. If the question asks about a specific event or fact, look for direct evidence in the memories.
   4. Your answer must be grounded in the memories. However, you may use general world knowledge to interpret or complete information found within a memory (e.g., identifying a landmark mentioned by description).
   5. If the question involves time references (like "last year", "two months ago", etc.), you **must** calculate the actual date based on the memory's timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
   6. Always convert relative time references to specific dates, months, or years in your final answer.
   7. Do not confuse character names mentioned in memories with the actual users who created them.
   8. Keep the answer concise and direct, with no extra description. For list, commonality, or multi-item questions, include all distinct supported items even if this exceeds 5-6 words.

   # APPROACH (Think step by step):
   1. First, examine all memories that contain information related to the question.
   2. Synthesize findings from multiple memories if a single entry is insufficient.
   3. Examine timestamps and content carefully, looking for explicit dates, times, locations, or events.
   4. If the answer requires calculation (e.g., converting relative time references), perform the calculation.
   5. Formulate a precise, concise answer based on the evidence from the memories (and allowed world knowledge).
   6. Double-check that your answer directly addresses the question asked and adheres to all instructions.
   7. Ensure your final answer is specific and avoids vague time references.

   {context}

   Question: {question}

   Answer:
   """

# -- LongMemEval --

LME_ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories.
    2. Pay special attention to the timestamps to determine the answer.
    3. If the question asks about a specific event or fact, look for direct evidence in the memories.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question.
    2. Examine the timestamps and content of these memories carefully.
    3. Look for explicit mentions of dates, times, locations, or events that answer the question.
    4. If the answer requires calculation (e.g., converting relative time references), show your work.
    5. Formulate a precise, concise answer based solely on the evidence in the memories.
    6. Double-check that your answer directly addresses the question asked.
    7. Ensure your final answer is specific and avoids vague time references.

    {context}

    Current Date: {question_date}

    Question: {question}

    Answer:
    """

# -- BEAM --

BEAM_ANSWER_PROMPT = """
You are a knowledgeable and helpful AI assistant with access to memory context about a user.

# CONTEXT:
The following are relevant memories retrieved from prior conversations with the user.

{context}

# INSTRUCTIONS:
1. Carefully analyze the provided memory context.
2. Answer the question as thoroughly and accurately as possible based on the memories.
3. If the memories contain specific details (dates, names, numbers, preferences), include them in your answer.
4. If the memories are insufficient to fully answer the question, provide the best answer you can with available information and note any gaps.
5. Be detailed and comprehensive in your response.

# QUESTION:
{question}

# ANSWER:
"""

# -- HaluMem --

HM_ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with answering questions based on retrieved conversation memories.

# CONTEXT:
You have access to memories from prior conversations with a user. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to timestamps and the chronological order of events.
3. Answer ONLY based on information present in the provided memories.
4. If the memories contain conflicting information, use the most recent one unless the question specifically asks about an earlier time.
5. If the memories do not contain sufficient information to answer the question, say so clearly.
6. Do NOT fabricate or hallucinate any information not present in the memories.

# APPROACH (Think step by step):
1. Identify which memories are relevant to the question.
2. Check for any updates or changes over time in the relevant memories.
3. Formulate a precise, concise answer grounded solely in the evidence from memories.
4. Double-check that your answer directly addresses the question asked.

{context}

Question: {question}

Answer:
"""


# -- PersonaMem v2 --

PM_ANSWER_PROMPT = """
    You are a helpful assistant tasked with selecting the best answer to a user question, based solely on summarized conversation memories.

    # CONTEXT:
    The following are summarized facts and preferences extracted from prior user conversations. Use only these memories to answer the question.

    {context}

    # INSTRUCTIONS:
    1. Carefully read and reason over the memory summary.
    2. Evaluate each of the four answer choices (a) through (d).
    3. Choose the single best-supported answer based on the information in memory.
    4. Output ONLY the final choice in the format (a), (b), (c), or (d), placed directly after the token <final_answer>.

    # IMPORTANT RULES:
    - Your final answer **must appear after** the token <final_answer>.
    - Your final answer **must use parentheses**, like (a) or (b).
    - Do NOT list multiple choices. Choose only one.
    - Do NOT include extra text after <final_answer>. Just output the answer.

    # QUESTION:
    {question}

    # OPTIONS:
    {options}

    Final Answer:
    <final_answer>
"""


# ── 4. Benchmark judge prompts ───────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions "
    "match a gold standard answer."
)

JUDGE_PROMPT = """
    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it's time for the real question:
    Question: {question}
    Gold answer: {golden_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """

HM_JUDGE_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given:
    (1) a question about a user's personal information, preferences, or experiences,
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The question tests whether a memory system can accurately recall and reason about user information
from prior conversations. The gold answer is the factually correct answer based on the conversation history.

Grading guidelines:
- Be generous: as long as the generated answer conveys the same core meaning as the gold answer, mark it CORRECT.
- For factual questions (names, dates, numbers, locations), the generated answer must match the gold answer's key facts.
- For questions about updates or changes, the answer must reflect the LATEST state unless the question asks about a prior state.
- If the generated answer says "I don't know" or "no information available" but the gold answer exists, mark it WRONG.
- If the generated answer fabricates information not in the gold answer, mark it WRONG.

Question: {question}
Gold answer: {golden_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


BEAM_RUBRIC_ITEM_JUDGE_PROMPT = """\
You are an expert evaluator tasked with judging whether the LLM's response \
demonstrates compliance with the specified RUBRIC CRITERION.

## QUESTION:
{question}

## RUBRIC CRITERION (what to check):
{rubric_item}

## RESPONSE TO EVALUATE:
{response}

## SCORING:
- **1.0** = Fully satisfied: The response clearly and completely addresses \
this rubric criterion.
- **0.5** = Partially satisfied: The response addresses this criterion but \
is incomplete, vague, or only partially correct.
- **0.0** = Not satisfied: The response does not address this criterion at \
all, or is incorrect.

## SEMANTIC TOLERANCE:
Judge by meaning, not exact wording. Accept paraphrases and synonyms that \
preserve intent. Numbers/currencies/dates may appear in equivalent forms \
(e.g. "$68,000", "68k"). Treat them as equal when numerically equivalent.

Evaluate the response against ONLY this specific rubric criterion.

Return your evaluation as JSON:
{{"score": <0.0 or 0.5 or 1.0>, "reason": "<brief explanation>"}}\
"""

BEAM_EVENT_ORDERING_JUDGE_PROMPT = """\
You are an expert evaluator. The question asked the user to list events in \
chronological order. The reference ordering of events is given below as a \
numbered list. Your task is to determine the order in which these same \
events appear in the response.

## QUESTION:
{question}

## REFERENCE ORDERING (correct chronological order):
{reference_ordering}

## RESPONSE TO EVALUATE:
{response}

## INSTRUCTIONS:
For each event in the reference list, determine its position (1-based) in \
the response. If an event is NOT mentioned in the response at all, assign \
position -1.

Return a JSON object mapping each event label to its detected position:
{{"positions": [{{"event": "<event label>", "position": <int or -1>}}, ...]}}\
"""
