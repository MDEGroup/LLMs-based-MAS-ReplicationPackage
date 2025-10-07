import asyncio
import pandas as pd
from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from metric.rouge import ROUGE
from prompt.prompt import (
    EXTRACTOR_PROMPT,
    OPTIMIZED_SUMMARIZER_PROMPT
)

class Evaluation:
    def __init__(self):
        self.extractor_agent = ExtractorAgent()
        self.summarizer_agent = SummarizerAgent()
        self.extractor_prompt = EXTRACTOR_PROMPT
        self.summarizer_prompt = OPTIMIZED_SUMMARIZER_PROMPT
        self.dic_results = []

    async def run(self, dataset: list[dict]):
        
        for i, data in enumerate(dataset):
            try:
                description = data["description"]
                readme = data["readme"]
                print(f"Data #{i}:\n- Description: {description}\n")

                #### Extractor Agent ####
                extracted_text = await self.extractor_agent.run(
                    prompt=self.extractor_prompt, readme_text=readme
                )
                print(f"Extracted text: {extracted_text}\n")

                #### Summarizer Agent ####
                about = await self.summarizer_agent.run(
                    prompt=self.summarizer_prompt, extracted_text=extracted_text
                )
                print(f"Generated About: {about}\n")
                
                rougeL_score = ROUGE().get_RougeL(string_1=about, string_2=description)
                rouge1_score = ROUGE().get_Rouge1(string_1=about, string_2=description)
                rouge2_score = ROUGE().get_Rouge2(string_1=about, string_2=description)
                print(f"Rouge1 Score: {rouge1_score}")
                print(f"Rouge2 Score: {rouge2_score}")
                print(f"RougeL Score: {rougeL_score}")
                
                # Store result
                result_save = {
                    "Description": description,
                    "Generated About": about,
                    "ROUGE-1": rouge1_score,
                    "ROUGE-2": rouge2_score,
                    "ROUGE-L": rougeL_score,
                }
                self.dic_results.append(result_save)

            except Exception as e:
                print(f"Error while running data {i}: {e}")
                continue

class Main:
    def __init__(self):
        pass

    def run(self):
        test_data = pd.read_csv("data-experiment/ES.csv").to_dict(orient="records")
        evaluation = Evaluation()
        asyncio.run(evaluation.run(test_data))
        df_results = pd.DataFrame(evaluation.dic_results)
        df_results.to_csv("results/evaluation_TS50.csv", index=False)

if __name__ == "__main__":
    main_flow = Main()
    main_flow.run()
