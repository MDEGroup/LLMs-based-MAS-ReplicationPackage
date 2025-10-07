import os
from agent.extractor import ExtractorAgent
from agent.summarizer import SummarizerAgent
from agent.evaluator import EvaluatorAgent
from agent.teacher import TeacherAgent
from agent.prompt_combine import PromptCombineAgent
from utils.rouge_plugin import RougePlugin
from utils.prompt_plugin import PromptPlugin
from utils.prompt_builder import PromptBuilder
from utils.agent_functions import AgentFunctions
from metric.rouge import ROUGE
from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion


class ParallelOptimizer:
    def __init__(self, threshold: float = 0.7):
        
        self.EXTRACTOR_TEMPLATE_FILE = "template/extractor.yaml"
        self.SUMMARIZER_TEMPLATE_FILE = "template/summarizer.yaml"
        self.EVALUATOR_TEMPLATE_FILE = "template/evaluator.yaml"
        self.TEACHER_TEMPLATE_FILE = "template/teacher.yaml"
        self.COMBINE_TEMPLATE_FILE = "template/prompt_combine.yaml"
        self.threshold = threshold
        
    def _create_kernel_with_chat_completion(self, service_id: str) -> Kernel:
        kernel = Kernel()
        kernel.add_service(OpenAIChatCompletion(service_id=service_id, api_key = os.getenv("OPENAI_API_KEY")))
        return kernel

    async def run(self, max_iterations: int, train_data: list[dict]):
        
        # Store different prompt from the interactions
        data_prompt = [] 
        EXTRACTOR_NAME = "Extractor"
        SUMMARIZER_NAME = "Summarizer"
        TEACHER_NAME = "Teacher"
        EVALUATOR_NAME = "Evaluator"
        COMBINE_NAME = "Combiner"
        
        for i, data in enumerate(train_data):   
            
            # Ground truth value
            description = data["description"]
            # Readme value
            readme = data["readme"]
            
            print(f"Data #{i}:\n- Description: {description}")
            
            # Create Extractor Agent
            extractor_agent_handler =  ExtractorAgent(EXTRACTOR_NAME)
            extractor_agent = extractor_agent_handler.create_agent(self.EXTRACTOR_TEMPLATE_FILE, readme)

            # Start Conversation Chat Extractor
            extracted_text = await extractor_agent.get_response(messages=None)
            extracted_text = extracted_text.content
            print(f"Extracted text: {extracted_text}")
            
            # Initialize plugins
            rouge_plugin = RougePlugin()
            prompt_plugin = PromptPlugin()
            
            # Create Agent Summarizer
            summarizer_agent_handler =  SummarizerAgent(SUMMARIZER_NAME)
            summarizer_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            summarizer_agent = summarizer_agent_handler.create_agent(self.SUMMARIZER_TEMPLATE_FILE, extracted_text)
            
            # Create Agent Evaluator 
            evaluator_agent_handler =  EvaluatorAgent(EVALUATOR_NAME)
            evaluator_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            evaluator_agent_handler.add_plugin_kernel("rouge_plugin", rouge_plugin)
            evaluator_agent = evaluator_agent_handler.create_agent(self.EVALUATOR_TEMPLATE_FILE, description)

            # Create Agent Teacher
            teacher_agent_handler =  TeacherAgent(TEACHER_NAME)
            teacher_agent_handler.add_plugin_kernel("prompt_plugin", prompt_plugin)
            teacher_agent_handler.add_plugin_kernel("rouge_plugin", rouge_plugin)
            teacher_agent = teacher_agent_handler.create_agent(self.TEACHER_TEMPLATE_FILE, ground_truth=description, extracted_text=extracted_text)

            # Initializing GroupChat
            kernel = self._create_kernel_with_chat_completion("kernel_loop") #Creating this kernel outside --> I dont know why it was not working inside AgentFunctions
            agent_functions =  AgentFunctions(kernel)    
            group_chat = AgentGroupChat(
                agents=[summarizer_agent,evaluator_agent,  teacher_agent],
                selection_strategy=agent_functions.get_selection_function(summarizer_agent, teacher_agent, evaluator_agent),
                termination_strategy=agent_functions.get_termination_function(evaluator_agent, description, max_iterations=45),
            )
            
            initial_summarizer_prompt = """
            Summarize the following extracted text from a Github repository README into a short term/phrase introducing the repository:
            
            The output should include only a short term/phrase introducing the repository.
            """
            # Initialize prompt in the plugin
            await prompt_plugin.set_last_instruction(initial_summarizer_prompt)
            
            # Start Conversation
            await group_chat.add_chat_message(message=initial_summarizer_prompt)
            print("\n\nGroupChat: Summarizer - Evaluator - Teacher\n")
            async for content in group_chat.invoke():
                print(f"# {content.name}: {content.content}\n")
            
            # Get best prompts
            last_summary = await prompt_plugin.get_last_summary()
            rouge_l = ROUGE().get_RougeL(
                    string_1=last_summary,
                    string_2=description
                )
            last_instruction = await prompt_plugin.get_last_instruction()
            if(rouge_l >= .7):
                print(f"Added Prompt #{i}: {last_instruction}")
                data_prompt.append(last_instruction)

            print(f"Length data_prompt: {len(data_prompt)}")
            
        # Call Prompt Combiner
        summarizer_list = PromptBuilder._clean_prompt_list(data_prompt)
        combine_agent_handler =  PromptCombineAgent(COMBINE_NAME)
        combine_agent = combine_agent_handler.create_agent(self.COMBINE_TEMPLATE_FILE, summarizer_list)
        
        # Generate the agent response
        combined_text = await combine_agent.get_response(messages=None)
        combined_text = combined_text.content.content
        print(f"Extracted text: {combined_text}")
            
        # Show history of all best prompts
        with open("results/best_prompts_TS50.txt", "w", encoding="utf-8") as f:
            for line in data_prompt:
                f.write(line + "\n")
        
        # Optimized prompt
        with open("results/optimized_prompt_TS50.txt", "w", encoding="utf-8") as f:
            f.write(combined_text)
            