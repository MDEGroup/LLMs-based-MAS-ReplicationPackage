import asyncio
import pandas as pd
from agent.extractor import ExtractorAgent
from agent.summarizer_evaluation import SummarizerEvaluationAgent
from metric.rouge import ROUGE
from dotenv import load_dotenv
from prompt.prompt import (
    OPTIMIZED_SUMMARIZER_PROMPT,
    EXTRACTOR_PROMPT
)

class Evaluation:
    def __init__(self):
        self.EXTRACTOR_NAME = "Extractor"
        self.SUMMARIZER_NAME = "Summarizer"
        self.dic_results = []

    async def run(self, dataset: list[dict]):
        
        for i, data in enumerate(dataset):
            try:
                description = data["description"]
                readme = data["readme"]
                print(f"Data #{i}:\n- Description: {description}\n")

                #### Extractor Agent ####
                extractor_agent =  ExtractorAgent(self.EXTRACTOR_NAME)
                extracted_text = await extractor_agent.run_agent(EXTRACTOR_PROMPT, readme)

                #### Summarizer Agent ####
                summarizer_agent =  SummarizerEvaluationAgent(self.SUMMARIZER_NAME)
                about = await summarizer_agent.run_agent(OPTIMIZED_SUMMARIZER_PROMPT, extracted_text)
                
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
                    "ROUGE-L": rougeL_score,
                    "ROUGE-1": rouge1_score,
                    "ROUGE-2": rouge2_score,
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
        df_results.to_csv("results/evaluation_TS10_third_try.csv", index=False)


if __name__ == "__main__":
    load_dotenv()
    main_flow = Main()
    main_flow.run()
