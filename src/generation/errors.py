class OrchestratorError(Exception):
    pass


class LLMError(OrchestratorError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMAPIError(LLMError):
    pass


class LLMFormatError(LLMError):
    pass


class RetrievalError(OrchestratorError):
    pass


class RetrievalTimeoutError(RetrievalError):
    pass


class RetrievalAPIError(RetrievalError):
    pass