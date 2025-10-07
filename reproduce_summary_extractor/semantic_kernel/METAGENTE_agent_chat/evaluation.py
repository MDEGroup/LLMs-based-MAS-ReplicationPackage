import asyncio
import pandas as pd
from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from metric.rouge import ROUGE

class Evaluation:
    def __init__(self):
        self.EXTRACTOR_TEMPLATE_FILE = "template/extractor.yaml"
        self.SUMMARIZER_TEMPLATE_FILE = "template/summarizer_best.yaml"
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
                extractor_agent_handler =  ExtractorAgent(self.EXTRACTOR_NAME)
                extractor_agent = extractor_agent_handler.create_agent(self.EXTRACTOR_TEMPLATE_FILE, readme)    
            
                # Get response Extractor
                extracted_text = await extractor_agent.get_response(messages=None)
                extracted_text = extracted_text.content
                print(f"Extracted text: {extracted_text}\n")

                #### Summarizer Agent ####
                summarizer_agent_handler =  SummarizerAgent(self.SUMMARIZER_NAME)
                summarizer_agent = summarizer_agent_handler.create_agent(self.SUMMARIZER_TEMPLATE_FILE, extracted_text)
            
                # Get response Summarizer
                summarized_text = await summarizer_agent.get_response(messages=None)
                about = summarized_text.content.content
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
