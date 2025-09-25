from typing import Dict

from langchain_core.chat_loaders import BaseChatLoader


class AquaRagLLMFactory:
    def __init__(self, cfg: Dict):
        self.cfg_retriever = cfg["models"]["retriever"]
        self.cfg_responder = cfg["models"]["responder"]
        self.cfg_multi_query = cfg["models"]["multi-query"]

    def create_retriever_model(self) -> BaseChatLoader:
        provider: str = self.cfg_retriever["provider"]
        name: str = self.cfg_retriever["name"]
        temperature: float = self.cfg_retriever["temperature"]

        try:
            model = self._create_model(provider, name, temperature)
        except Exception:
            raise ValueError("There was a problem while creating retriver model")

        return model

    def _create_model(
        self, provider: str, name: str, temperature: int
    ) -> BaseChatLoader:
        model = None

        if provider.lower() == "openai":
            from langchain_openai import ChatOpenAI

            try:
                model = ChatOpenAI(model=name, temperature=temperature)
            except Exception as e:
                raise ValueError(
                    f"There was a problem while creating OpenAi llm model {name}: {e}"
                )

        elif provider.lower() == "mistral":
            from langchain_mistralai import ChatMistralAI

            try:
                model = ChatMistralAI(model=name, temperature=temperature)
            except Exception as e:
                raise ValueError(
                    f"There was a problem while creating Mistral llm model {name}: {e}"
                )

        else:
            raise ValueError(f"Unkown llm provider: {provider}")

        return model
