EXTRACTOR_PROMPT = """
Your task is to shorten and extract only the introduction and description information from the README of a Github repository. You are given the following README text from a GitHub repository:
<README>
$readme_text
</README>
 
# Steps
- **Identify the structure of the repository**: The README file is a structure text file that might contains many sections such as introduction, description, installation, contributing, license,...
- **Remove all sections that are not relevant to the introduction or description of the repository**: Irrelevant sections might include technical guidance (installing/running/specification... instruction), repository structure/table of contents, contributions/references,...
- **Remove all unnecessary links/tags**: Identify all links/tags that DO NOT contribute to the description of the repository. You must remove all of these reference links and tags.
- **Return only text that is relevant to the description of the repository**: The output should only contains the text that is relevant to the introduction/description of the repository, including the project name/title, project tagline/functional description/purpose statement/overview. DO NOT include any output identifications such as: "Here's the ..." or "Extracted README:"
"""

INITIAL_SUMMARIZER_PROMPT = """
Summarize the following extracted text from a Github repository README into a short term/phrase introducing the repository:
<EXTRACTED_README>
$extracted_text
</EXTRACTED_README>
 
The output should include only a short term/phrase introducing the repository.
"""

TEACHER_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the README of a Github repository. Your task is to modify and improve the current prompt of the LLM based on the result of testing on a data include a README and a ground truth description.
 
# Steps:
- **Analyze the data for testing**: Analyze the following data include an extracted text from a README and a ground truth description from a GitHub repository:
<EXTRACTED_TEXT>
$extracted_text
</EXTRACTED_TEXT>
 
<GROUND_TRUTH DESCRIPTION>
$description
</GROUND_TRUTH DESCRIPTION>
- **Review the current result**: Review the generated description using the extracted text its ROUGE score on the ground truth description to identify improvements that could be made:
<GENERATED_DESCRIPTION>
$generated_about
</GENERATED_DESCRIPTION>
<ROUGE_SCORE>
$rouge_score
</ROUGE_SCORE>
- **Prioritize extracting existing tagline/functional description/purpose statement/overview**: Compare the text from the beginning of the extracted text from README and the ground truth description. If the ground truth description is already existed in this extracted text as a tagline/functional description/purpose statement/overview, you must include in the new prompt the instruction to prioritize using it.
- **Modify the current prompt**: Identify mistakes and lacking instructions in the current prompt from the result of the above review. You should preserve the current prompt as much as possible and only make small changes to the prompt based on the identified mistakes and lacking instructions.
<CURRENT_PROMPT>
$summarizer_prompt
</CURRENT_PROMPT>
As the new prompt will not include the ground truth description, DO NOT mention about the ground truth description in the new prompt. DO NOT include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the new prompt for the LLM
"""

COMBINE_PROMPT = """
You are a professional Prompt Engineer. You are working on a system using a Large Language Model (LLM) to help developers automatically generate a short Description term/phrase contain key concept/idea from an extracted text of the README of a Github repository. Your task is to combine several candidate prompts for the LLM into a final prompt.
 
# Steps:
- **Review all candidate prompts**: Analyze the following prompts to identify common parts to be included in the final prompt and also includes specific details or conditional key points from these prompts to be included in the final prompt
<CANDIDATE_PROMPTS>
$summarizer_list
</CANDIDATE_PROMPTS>
- **Generate a final prompt**: Based on the common parts and conditional key points, generate a final prompt for the LLM.

# Output Format:
Do not include any reasoning/explanation like "Based on the result of the above review:", "Here's the", ... or any output identifiers like "Prompt:", "New Prompt", ... The output should only include a string representing the prompt for the LLM
"""


OPTIMIZED_SUMMARIZER_PROMPT = """Summarize the following extracted text from a GitHub repository README into a concise term or phrase introducing the repository. If a tagline, functional description, purpose statement, or overview is present at the beginning of the text, prioritize using it verbatim as the description. Ensure the output captures the key concept or idea of the repository, reflecting any specific terms or distinctive elements present in the extracted text. The output should include only a short term or phrase introducing the repository.

<EXTRACTED_README>
$extracted_text
</EXTRACTED_README>"""
